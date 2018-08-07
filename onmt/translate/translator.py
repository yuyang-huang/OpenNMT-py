#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import argparse
import codecs
import os
import math

import torch

from itertools import count
from tqdm import tqdm

from onmt.utils.misc import tile
from onmt.modules import SharedVocabCopyGenerator
import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.opts as opts


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model, model_opt = \
        onmt.model_builder.load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)

    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat", "src_seq_length_trunc",
                        "ignore_when_blocking", "dump_beam", "report_bleu", "report_rouge",
                        "data_type", "replace_unk", "gpu", "verbose", "fast"]}

    translator = Translator(model, fields, global_scorer=scorer,
                            out_file=out_file, report_score=report_score,
                            copy_attn=model_opt.copy_attn, logger=logger,
                            **kwargs)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 model,
                 fields,
                 beam_size,
                 n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 logger=None,
                 gpu=False,
                 dump_beam="",
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[],
                 sample_rate='16000',
                 window_size=.02,
                 window_stride=.01,
                 window='hamming',
                 use_filter_pred=False,
                 data_type="text",
                 replace_unk=False,
                 report_score=True,
                 report_bleu=False,
                 report_rouge=False,
                 verbose=False,
                 out_file=None,
                 fast=False,
                 src_seq_length_trunc=0):
        self.logger = logger
        self.gpu = gpu
        self.cuda = gpu > -1

        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.use_filter_pred = use_filter_pred
        self.replace_unk = replace_unk
        self.data_type = data_type
        self.verbose = verbose
        self.out_file = out_file
        self.report_score = report_score
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.fast = fast
        self.src_seq_length_trunc = src_seq_length_trunc

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate(self,
                  src_path=None,
                  src_data_iter=None,
                  tgt_path=None,
                  tgt_data_iter=None,
                  src_dir=None,
                  batch_size=None,
                  attn_debug=False):
        """
        Translate content of `src_data_iter` (if not None) or `src_path`
        and get gold scores if one of `tgt_data_iter` or `tgt_path` is set.

        Note: batch_size must not be None
        Note: one of ('src_path', 'src_data_iter') must not be None

        Args:
            src_path (str): filepath of source data
            src_data_iter (iterator): an interator generating source data
                e.g. it may be a list or an openned file
            tgt_path (str): filepath of target data
            tgt_data_iter (iterator): an interator generating target data
            src_dir (str): source directory path
                (used for Audio and Image datasets)
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        assert src_data_iter is not None or src_path is not None

        if batch_size is None:
            raise ValueError("batch_size must be set")
        data = inputters.build_dataset(self.fields,
                                       self.data_type,
                                       src_path=src_path,
                                       src_data_iter=src_data_iter,
                                       tgt_path=tgt_path,
                                       tgt_data_iter=tgt_data_iter,
                                       src_dir=src_dir,
                                       src_seq_length_trunc=self.src_seq_length_trunc,
                                       sample_rate=self.sample_rate,
                                       window_size=self.window_size,
                                       window_stride=self.window_stride,
                                       window=self.window,
                                       use_filter_pred=self.use_filter_pred)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        builder = onmt.translate.TranslationBuilder(
            data, self.fields,
            self.n_best, self.replace_unk, tgt_path)

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        for batch in tqdm(data_iter):
            batch_data = self.translate_batch(batch, data, fast=self.fast)
            translations = builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt_path is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                all_predictions += [n_best_preds]
                self.out_file.write('\n'.join(n_best_preds) + '\n')
                self.out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode('utf-8'))

                # Debug attention.
                if attn_debug:
                    srcs = trans.src_raw
                    preds = trans.pred_sents[0]
                    preds.append('</s>')
                    attns = trans.attns[0].tolist()
                    header_format = "{}" + "|||{}" * len(srcs)
                    row_format = "{}" + "|||{:.7f}" * len(srcs)
                    output = header_format.format("", *trans.src_raw) + '\n'
                    for word, row in zip(preds, attns):
                        max_index = row.index(max(row))
                        row_format = row_format.replace(
                            "|||{:.7f}", "|||*{:.7f}", max_index + 1)
                        row_format = row_format.replace(
                            "|||*{:.7f}", "|||{:.7f}", max_index)
                        output += row_format.format(word, *row) + '\n'
                        row_format = "{}" + "|||{:.7f}" * len(srcs)
                    os.write(1, output.encode('utf-8'))

        if self.report_score:
            msg = self._report_score('PRED', pred_score_total,
                                     pred_words_total)
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            if tgt_path is not None:
                msg = self._report_score('GOLD', gold_score_total,
                                         gold_words_total)
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)
                if self.report_bleu:
                    msg = self._report_bleu(tgt_path)
                    if self.logger:
                        self.logger.info(msg)
                    else:
                        print(msg)
                if self.report_rouge:
                    msg = self._report_rouge(tgt_path)
                    if self.logger:
                        self.logger.info(msg)
                    else:
                        print(msg)

        if self.dump_beam:
            import json
            json.dump(self.translator.beam_accum,
                      codecs.open(self.dump_beam, 'w', 'utf-8'))
        return all_scores, all_predictions

    def translate_batch(self, batch, data, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            if fast:
                return self._fast_translate_batch(
                    batch,
                    data,
                    self.max_length,
                    min_length=self.min_length,
                    return_attention=self.replace_unk)
            else:
                return self._translate_batch(batch, data)

    def _fast_translate_batch(self,
                              batch,
                              data,
                              max_length,
                              min_length=0,
                              return_attention=False):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert data.data_type == 'text'
        assert not self.dump_beam
        assert not self.use_filter_pred

        beam_size = self.beam_size
        batch_size = batch.batch_size
        vocab = self.fields["tgt"].vocab
        start_token = vocab.stoi[inputters.BOS_WORD]
        end_token = vocab.stoi[inputters.EOS_WORD]
        pad_token = self.fields['src'].vocab.stoi[inputters.PAD_WORD]

        # Encoder forward.
        src = inputters.make_features(batch, 'src', data.data_type)
        _, src_lengths = batch.src
        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
            src, memory_bank, enc_states, with_cache=True)

        # Use variables to hold these attributes since batch size shrinks as search progresses
        src_map = batch.src_map
        src_indices = batch.indices
        src_mask = src.eq(pad_token).transpose(0, 1).view(batch_size, 1, -1)

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        memory_bank = tile(memory_bank, beam_size, dim=1)
        memory_lengths = tile(src_lengths, beam_size)

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=memory_bank.device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=memory_bank.device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            start_token,
            dtype=torch.long,
            device=memory_bank.device)
        alive_attn = None

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=memory_bank.device).repeat(batch_size))

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["attention"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = self._run_target(batch, data)
        results["batch"] = batch

        max_length += 1

        def _reorder(x, dim1, dim2):
            return (x.view(dim1, dim2, -1)
                    .transpose(0, 1)
                    .contiguous()
                    .view(dim1 * dim2, -1))

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1, 1)

            # Decoder forward.
            dec_out, dec_states, attn = self.model.decoder(
                decoder_input,
                memory_bank,
                dec_states,
                memory_lengths=memory_lengths,
                step=step,
                src=src)
            dec_out = dec_out.squeeze(0)

            # Generator forward.
            if self.copy_attn:
                vocab_size = len(vocab)
                if isinstance(self.model.generator, SharedVocabCopyGenerator):
                    out = self.model.generator.forward(
                        _reorder(dec_out, batch_size, beam_size),
                        _reorder(attn['copy'].squeeze(0), batch_size, beam_size),
                        src)
                    log_probs = _reorder(out.log(), beam_size, batch_size)
                else:
                    out = self.model.generator.forward(
                        _reorder(dec_out, batch_size, beam_size),
                        _reorder(attn['copy'].squeeze(0), batch_size, beam_size),
                        src_map)
                    probs = inputters.TextDataset.collapse_copy_scores(
                        out.data.view(beam_size, batch_size, -1),
                        src_indices,
                        vocab,
                        data.src_vocabs)
                    log_probs = _reorder(
                        probs.narrow(-1, 0, vocab_size).view(-1, vocab_size).log(),
                        beam_size, batch_size)
            else:
                log_probs = self.model.generator.forward(dec_out)
                vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            # Prune the completed hypotheses
            log_probs = torch.where(decoder_input.ne(end_token).squeeze(0),
                                    log_probs,
                                    torch.full_like(log_probs, -1e20))

            # Block ngram repeats
            if step > self.block_ngram_repeat > 0:
                # Take the trailing n-gram as the target for sliding window matching
                match_target = alive_seq[:, -self.block_ngram_repeat:]
                for j in range(1, step + 1 - self.block_ngram_repeat):
                    # If `window` matches `match_target`, prune the hypothesis
                    window = alive_seq[:, j:j + self.block_ngram_repeat]
                    match = (window - match_target).abs().sum(1).eq(0)
                    log_probs[match] = -1e20

            # Coverage penalty
            if alive_attn is not None:
                beta = self.global_scorer.beta
                cov = alive_attn.sum(0) + attn['std'].squeeze(0)
                coverage_penalty = torch.max(cov, torch.ones_like(cov))

                # Ignore penalty at padding positions
                coverage_penalty.view(batch_size, beam_size, -1).masked_fill_(src_mask, 0)
                coverage_penalty = beta * (coverage_penalty.sum(1) - memory_lengths.float())
                log_probs -= coverage_penalty.unsqueeze(1)

            # Length penalty
            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            log_probs /= length_penalty

            # Flatten probs into a list of possibilities.
            log_probs = log_probs.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = log_probs.topk(beam_size, dim=-1)

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                topk_beam_index
                + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty
            if alive_attn is not None:
                select_indices = batch_index.view(-1)
                coverage_penalty = (coverage_penalty
                                    .index_select(0, select_indices)
                                    .view(batch_size, beam_size))
                topk_log_probs += coverage_penalty

            # Allow partial sentence if reached `max_length`
            if step + 1 == max_length:
                finished = torch.ones_like(topk_ids)
            else:
                finished = topk_ids.eq(end_token)

            # Save result of finished sentences.
            finished_indices = finished.nonzero()
            num_finished = len(finished_indices)
            if num_finished > 0:
                # Reorder before output
                select_indices = batch_index.view(-1)
                predictions = (alive_seq
                               .index_select(0, select_indices)
                               .view(-1, beam_size, alive_seq.size(-1)))
                if alive_attn is not None:
                    attention = (alive_attn
                                 .index_select(1, select_indices)
                                 .view(alive_attn.size(0), -1, beam_size, alive_attn.size(-1)))
                else:
                    attention = None
                scores = topk_scores.view(-1, beam_size)  # score is in correct order

                for n in range(num_finished):
                    i, j = finished_indices[n].tolist()
                    b = batch_offset[i]
                    results["predictions"][b].append(predictions[i, j, 1:])
                    results["scores"][b].append(scores[i, j])
                    if not return_attention:
                        results["attention"][b].append([])
                    else:
                        results["attention"][b].append(
                            attention[:, i, j, :memory_lengths[i]])

            # If all batches are finished, no need to go further.
            non_finished = finished.sum(-1).lt(beam_size).nonzero().view(-1)
            if len(non_finished) == 0:
                break

            # Remove finished batches for the next step.
            batch_size = len(non_finished)
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            topk_ids = topk_ids.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            src = src.index_select(1, non_finished)
            src_map = src_map.index_select(1, non_finished)
            src_indices = src_indices.index_select(0, non_finished)
            src_mask = src_mask.index_select(0, non_finished)

            # Select and reorder inputs to the next step
            select_indices = batch_index.view(-1)
            alive_seq = alive_seq.index_select(0, select_indices)
            memory_bank = memory_bank.index_select(1, select_indices)
            memory_lengths = memory_lengths.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

            if return_attention or self.global_scorer.beta > 0:
                # Update attention only if returning it or using coverage
                if alive_attn is None:
                    alive_attn = attn['std']
                else:
                    alive_attn = torch.cat([alive_attn, attn['std']], 0)
                alive_attn = alive_attn.index_select(1, select_indices)

            # Append last prediction.
            alive_seq = torch.cat([alive_seq, topk_ids.view(-1, 1)], -1)

        # Sort the extracted full hypotheses according to score
        for b in range(batch.batch_size):
            scores = results["scores"][b]
            sorting_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
            for key in ('predictions', 'scores', 'attention'):
                results[key][b] = [results[key][b][i] for i in sorting_indices]

        return results

    def _translate_batch(self, batch, data):
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[inputters.PAD_WORD],
                                    eos=vocab.stoi[inputters.EOS_WORD],
                                    bos=vocab.stoi[inputters.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return torch.tensor(a, requires_grad=False)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = inputters.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src

        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
            src, memory_bank, enc_states)

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data)\
                                                  .long()\
                                                  .fill_(memory_bank.size(0))

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        memory_bank = rvar(memory_bank.data)
        memory_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, dec_states, attn = self.model.decoder(
                inp, memory_bank, dec_states,
                memory_lengths=memory_lengths,
                step=i, src=src)

            dec_out = dec_out.squeeze(0)

            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
                beam_attn = unbottle(attn["std"])
            else:
                if isinstance(self.model.generator, SharedVocabCopyGenerator):
                    out = self.model.generator.forward(dec_out,
                                                       attn["copy"].squeeze(0),
                                                       src)
                    out = unbottle(out)
                else:
                    out = self.model.generator.forward(dec_out,
                                                       attn["copy"].squeeze(0),
                                                       src_map)
                    # beam x (tgt_vocab + extra_vocab)
                    out = data.collapse_copy_scores(
                        unbottle(out.data),
                        batch.indices, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()
                beam_attn = unbottle(attn["copy"])

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j],
                          beam_attn.data[:, j, :memory_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch

        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = inputters.make_features(batch, 'src', data_type)
        tgt_in = inputters.make_features(batch, 'tgt')[:-1]

        vocab = self.fields["tgt"].vocab
        index = batch.indices.data

        #  (1) run the encoder on the src
        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = \
            self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        dec_out, _, attn = self.model.decoder(
            tgt_in, memory_bank, dec_states, memory_lengths=src_lengths, src=src)

        # default for models without copy
        copy_attn = attn.get('copy', [[0]] * len(dec_out))

        tgt_pad = self.fields["tgt"].vocab.stoi[inputters.PAD_WORD]
        for dec, tgt, copy_prob in zip(dec_out, batch.tgt[1:].data, copy_attn):
            # Log prob of each word.
            if self.copy_attn:
                if isinstance(self.model.generator, SharedVocabCopyGenerator):
                    out = self.model.generator.forward(dec, copy_prob, src).log()
                else:
                    out = self.model.generator.forward(
                        dec,
                        copy_prob,
                        batch.src_map)
                    vocab_size = len(vocab)
                    probs = inputters.TextDataset.collapse_copy_scores(
                        out.unsqueeze(0),
                        index,
                        vocab,
                        data.src_vocabs)
                    out = probs.narrow(-1, 0, vocab_size).view(-1, vocab_size).log()
            else:
                out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores.view(-1)
        return gold_scores

    def _report_score(self, name, score_total, words_total):
        msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
            name, score_total / words_total,
            name, math.exp(-score_total / words_total)))
        return msg

    def _report_bleu(self, tgt_path):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")
        # Rollback pointer to the beginning.
        self.out_file.seek(0)
        print()

        res = subprocess.check_output("perl %s/tools/multi-bleu.perl %s"
                                      % (base_dir, tgt_path),
                                      stdin=self.out_file,
                                      shell=True).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_rouge(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        res = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN"
            % (path, tgt_path),
            shell=True,
            stdin=self.out_file).decode("utf-8")
        msg = res.strip()
        return msg
