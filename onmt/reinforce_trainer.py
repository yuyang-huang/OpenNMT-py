import sys
import time
import pickle

import torch
import torch.nn as nn
import sentencepiece as spm
from sumeval.metrics.rouge import RougeCalculator

import onmt.inputters as inputters
from onmt.utils import Statistics, loss
from onmt.utils.logging import logger
from onmt.modules import SharedVocabCopyGenerator
from onmt.trainer import Trainer


sp = spm.SentencePieceProcessor()
rouge = RougeCalculator(lang='zh')


def _reward(pred, references, char_based=False):
    """Compute ROUGE.

    Parameters
    ----------
    pred : str
        Predicted summary as a list of tokens.
    references : List[str]
        Reference summary or list of summaries.

    Returns
    -------
    The cluster-based ROUGE-W F1 score.
    """
    if char_based:
        pred = sp.DecodePieces([x.encode('utf8') for x in pred])
        references = [
            sp.DecodePieces([x.encode('utf8') for x in ref])
            for ref in references
        ]
    else:
        pred = ' '.join(pred)
        references = [' '.join(ref) for ref in references]
    return rouge.rouge_w(summary=pred, references=references)


def reward_worker(args):
    batch_idx, references, baseline_pred, sampled_pred = args
    if sampled_pred is None:
        # eval mode
        return _reward(baseline_pred, references)
    return batch_idx, _reward(baseline_pred, references), _reward(sampled_pred, references)


class ReinforceTrainer(Trainer):
    def __init__(self, model, train_loss, valid_loss, optim,
                 shard_size=32, data_type='cluster',
                 report_manager=None, model_saver=None):
        assert data_type == 'cluster'
        assert isinstance(model.generator, SharedVocabCopyGenerator)

        super().__init__(
            model, train_loss, valid_loss, optim,
            shard_size=shard_size,
            data_type=data_type,
            report_manager=report_manager,
            model_saver=model_saver,
        )

    def train(self, train_iter_fct, valid_iter_fct, train_steps, valid_steps):
        logger.info('Start training...')

        step = self.optim._step + 1
        train_iter = train_iter_fct()

        self._start_report_manager()

        while step <= train_steps:
            for batch in train_iter:
                report_stats = ReinforceStatistics()
                gamma = self.train_loss.gamma

                # forward and backward prop
                self.model.zero_grad()
                batch_loss, batch_stats = self._compute_loss(batch, gamma)
                batch_loss.div(batch.batch_size).backward()
                self.optim.step()

                # report and saving
                report_stats.update(batch_stats, update_n_src_words=True)
                self._maybe_report_training(
                    step, train_steps,
                    self.optim.learning_rate,
                    report_stats)

                if step % valid_steps == 0:
                    valid_iter = valid_iter_fct()
                    valid_stats = self.validate(valid_iter)
                    self._report_step(self.optim.learning_rate,
                                      step, valid_stats=valid_stats)

                self._maybe_save(step)
                step += 1
                if step > train_steps:
                    break

            train_iter = train_iter_fct()

    def validate(self, valid_iter):
        # Set model in validating mode.
        self.model.eval()

        stats = ReinforceStatistics()
        for batch in valid_iter:
            src = inputters.make_features(batch, 'src', self.data_type)
            _, src_lengths = batch.src
            tgt = inputters.make_features(batch, 'tgt')

            enc_states, memory_bank = self.model.encoder(src, src_lengths)
            dec_states = self.model.decoder.init_decoder_state(src, memory_bank, enc_states)
            baseline_preds, _ = self._sample(
                src, src_lengths, memory_bank, dec_states, max_length=tgt.size(0), greedy=True)

            batch_stats = self.valid_loss.rl_eval(batch, batch.tgt[1:], baseline_preds)
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()
        return stats

    def _compute_loss(self, batch, gamma):
        src = inputters.make_features(batch, 'src', self.data_type)
        _, src_lengths = batch.src
        tgt = inputters.make_features(batch, 'tgt')
        enc_states, memory_bank = self.model.encoder(src, src_lengths)

        # for ML loss: teacher forcing
        dec_states = self.model.decoder.init_decoder_state(src, memory_bank, enc_states)
        output, _, attns = self.model.decoder(tgt[:-1], memory_bank, dec_states,
                                              memory_lengths=src_lengths)
        ml_loss, ml_stats = self.train_loss.ml_loss(
            batch, output, src, batch.tgt[1:], attns['copy'])
        ml_stats.n_src_words += src_lengths.sum().item()

        if gamma == 0:
            return ml_loss, ml_stats

        # for RL loss: sample sequence versus greedy search sequence
        with torch.no_grad():
            dec_states = self.model.decoder.init_decoder_state(src, memory_bank, enc_states)
            baseline_preds, _ = self._sample(
                src, src_lengths, memory_bank, dec_states, max_length=tgt.size(0), greedy=True)

        dec_states = self.model.decoder.init_decoder_state(src, memory_bank, enc_states)
        sampled_preds, sampled_prob = self._sample(
            src, src_lengths, memory_bank, dec_states, max_length=tgt.size(0))

        rl_loss, rl_stats = self.train_loss.rl_loss(
            batch, batch.tgt[1:], baseline_preds, sampled_preds, sampled_prob)
        rl_stats.merge(ml_stats, update_n_src_words=True)

        total_loss = (1 - gamma) * ml_loss + gamma * rl_loss
        return total_loss, rl_stats

    def _sample(self, src, memory_lengths, memory_bank, dec_states, max_length, greedy=False):
        vocab = self.train_loss.tgt_vocab
        start_token = vocab.stoi[inputters.BOS_WORD]
        end_token = vocab.stoi[inputters.EOS_WORD]
        pad_token = vocab.stoi[inputters.PAD_WORD]

        batch_size = memory_lengths.size(0)
        pred = torch.full(
            (batch_size,),
            start_token,
            dtype=torch.long,
            device=memory_bank.device)
        pred_mask = torch.full_like(pred, pad_token)
        finished = torch.full(
            (batch_size,),
            0,
            dtype=torch.uint8,
            device=memory_bank.device)
        finished_mask = torch.ones_like(finished)

        sampled_seq = [[] for _ in range(batch_size)]
        total_loss = 0.
        for step in range(max_length - 1):
            # Decoder forward.
            decoder_input = pred.view(1, batch_size, 1)
            dec_out, dec_states, attn = self.model.decoder(
                decoder_input,
                memory_bank,
                dec_states,
                memory_lengths=memory_lengths)
            dec_out = dec_out.squeeze(0)
            attn = attn['copy'].squeeze(0)

            # Sample next word.
            probs = self.model.generator.forward(dec_out, attn, src)
            if greedy:
                pred = torch.argmax(probs, -1)
            else:
                pred = torch.distributions.Categorical(probs).sample()

            # Mask the loss of already finished samples
            pred = torch.where(finished, pred_mask, pred)
            total_loss = total_loss + self.train_loss.criterion(probs.log(), pred)

            # Save to output
            for b, w in enumerate(pred.tolist()):
                if w != pad_token and w != end_token:
                    sampled_seq[b].append(vocab.itos[w])

            # Handle sequence ending
            finished = torch.where(pred.eq(end_token), finished_mask, finished)
            if finished.all():
                break

        return sampled_seq, total_loss


class ReinforceLossCompute(loss.LossComputeBase):
    def __init__(self, generator, tgt_vocab, normalize_by_length,
                 gamma, spm_model, cluster_path=None):
        super().__init__(generator, tgt_vocab)
        self.gamma = gamma
        self.pool = None

        # We lazily load datasets when there are more than one, so postpone
        # the setting of cur_dataset.
        self.cur_dataset = None
        self.normalize_by_length = normalize_by_length

        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, reduction='none')

        # For reward calculation
        sp.Load(spm_model)
        self.cluster2refs = self.load_cluster(cluster_path)

    def load_cluster(self, cluster_path):
        if not cluster_path:
            return {}

        with open(cluster_path, 'rb') as f:
            return pickle.load(f)

    def ml_loss(self, batch, output, source, target, copy_attn):
        target = target.view(-1)
        scores = self.generator(self._bottle(output),
                                self._bottle(copy_attn),
                                source)
        nll = self.criterion(scores.log(), target)

        # Compute sum of perplexities for stats
        loss_data = nll.sum().data.clone()
        stats = self._stats(loss_data, scores.data, target.data)

        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            pad_ix = batch.dataset.fields['tgt'].vocab.stoi[inputters.PAD_WORD]
            tgt_lens = batch.tgt.ne(pad_ix).float().sum(0)
            nll = nll.view(-1, batch.batch_size).sum(0)
            nll = torch.div(nll, tgt_lens).sum()
        else:
            nll = nll.sum()
        return nll, stats

    def rl_loss(self, batch, target, baseline_preds, sampled_preds, sampled_nll):
        clusters = batch.cluster.tolist()

        tasks = []
        for b, (baseline_pred, sampled_pred, cluster) in enumerate(zip(
                baseline_preds, sampled_preds, clusters)):
            if cluster not in self.cluster2refs:
                tgt = []
                for i in target[:, b].tolist():
                    w = self.tgt_vocab.itos[i]
                    if w == inputters.EOS_WORD:
                        break
                    tgt.append(w)
                references = [tgt]
            else:
                references = self.cluster2refs[cluster]
            tasks.append((b, references, baseline_pred, sampled_pred))

        loss = 0.
        total_baseline_rouge = 0.
        total_sampled_rouge = 0.
        for b, baseline_rouge, sampled_rouge in self.pool.imap_unordered(reward_worker, tasks):
            loss = loss + (sampled_rouge - baseline_rouge) * sampled_nll[b]
            total_baseline_rouge += baseline_rouge
            total_sampled_rouge += sampled_rouge

        stats = ReinforceStatistics(batch.batch_size,
                                    total_baseline_rouge,
                                    total_sampled_rouge,
                                    loss.item())
        return loss, stats

    def rl_eval(self, batch, target, baseline_preds):
        clusters = batch.cluster.tolist()

        tasks = []
        for b, (baseline_pred, cluster) in enumerate(zip(baseline_preds, clusters)):
            if cluster not in self.cluster2refs:
                tgt = []
                for i in target[:, b].tolist():
                    w = self.tgt_vocab.itos[i]
                    if w == inputters.EOS_WORD:
                        break
                    tgt.append(w)
                references = [tgt]
            else:
                references = self.cluster2refs[cluster]
            tasks.append((b, references, baseline_pred, None))

        baseline_rouge = sum(r for r in self.pool.imap_unordered(reward_worker, tasks))
        return ReinforceStatistics(batch.batch_size, baseline_rouge)


class ReinforceStatistics(Statistics):
    def __init__(self, n_sentences=0, baseline_rouge=0, sampled_rouge=0, rl_loss=0):
        super().__init__()
        self.n_sentences = n_sentences
        self.baseline_rouge = baseline_rouge
        self.sampled_rouge = sampled_rouge
        self.rl_loss = rl_loss

    def merge(self, stats, update_n_src_words=False):
        super().update(stats, update_n_src_words=update_n_src_words)

    def update(self, stats, update_n_src_words=False):
        self.merge(stats, update_n_src_words)
        self.rl_loss += stats.rl_loss
        self.baseline_rouge += stats.baseline_rouge
        self.sampled_rouge += stats.sampled_rouge
        self.n_sentences += stats.n_sentences

    def rouge(self, baseline=True):
        r = self.baseline_rouge if baseline else self.sampled_rouge
        return 100 * r / self.n_sentences

    def output(self, step, num_steps, learning_rate, start):
        t = self.elapsed_time()
        logger.info("Step %2d/%5d; acc: %5.2f; ppl: %5.2f; xent: %4.2f; "
                    "rg_s: %5.2f; rg_b: %5.2f; rll: %4.2f; "
                    "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec",
                    step, num_steps, self.accuracy(), self.ppl(), self.xent(),
                    self.rouge(baseline=False), self.rouge(),
                    self.rl_loss / self.n_sentences,
                    learning_rate, self.n_src_words / (t + 1e-5),
                    self.n_words / (t + 1e-5), time.time() - start)
        sys.stdout.flush()
