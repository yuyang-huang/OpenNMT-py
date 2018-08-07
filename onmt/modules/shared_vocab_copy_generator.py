""" Generator module """
import torch.nn as nn
import torch
import torch.cuda

import onmt.inputters as inputters
from onmt.utils import loss, Statistics
from onmt.modules.embeddings import TiedEmbeddingLinear


class SharedVocabCopyGenerator(nn.Module):
    """Copy generator with shared vocabulary (faster code path).
    """
    def __init__(self, input_size, tgt_dict, tied_embeddings):
        super().__init__()
        self.linear = TiedEmbeddingLinear(input_size, tied_embeddings)
        self.linear_copy = nn.Linear(input_size, 1)
        self.tgt_dict = tgt_dict
        self.pad_word = tgt_dict.stoi[inputters.PAD_WORD]
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden, attn, src):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.

        Args:
            hidden (`FloatTensor`): hidden outputs `[tlen * batch, input_size]`
            attn (`FloatTensor`): attention over source tokens `[tlen * batch, slen]`
            src (`LongTensor`): integer indices of each source word `[slen, batch, 1]`
        """
        # Probability of copying p(z=1)
        p_copy = self.sigmoid(self.linear_copy(hidden))

        # Original distribution
        logits = self.linear(hidden)
        logits[:, self.pad_word] = -float('inf')
        prob = self.softmax(logits) + 1e-20

        # Weighted by (1 - p_z) and reshape to [tlen, batch_size * tvocab] for index_add
        tvocab = prob.size(-1)
        batch_size = src.size(1)
        tlen = hidden.size(0) // batch_size
        prob = ((1 - p_copy) * prob).view(tlen, -1)

        # Add copy distribution
        batch_offset = torch.arange(0, batch_size * tvocab,
                                    step=tvocab,
                                    device=src.device).unsqueeze(1)
        src = (src.transpose(0, 1).squeeze(-1) + batch_offset).contiguous().view(-1)
        attn = (p_copy * attn).view(tlen, -1)
        return prob.index_add(-1, src, attn).view(tlen * batch_size, tvocab)


class SharedVocabCopyGeneratorLossCompute(loss.LossComputeBase):
    def __init__(self, generator, tgt_vocab, normalize_by_length):
        super().__init__(generator, tgt_vocab)

        # We lazily load datasets when there are more than one, so postpone
        # the setting of cur_dataset.
        self.cur_dataset = None
        self.normalize_by_length = normalize_by_length

        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, reduction='none')

    def _make_shard_state(self, batch, output, range_, attns):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "copy_attn": attns["copy"],
        }

    def _compute_loss(self, batch, output, source, target, copy_attn):
        """
        Compute the loss. The args must match self._make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            source: the source word indices.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
        """
        target = target.view(-1)
        scores = self.generator(self._bottle(output),
                                self._bottle(copy_attn),
                                source)
        loss = self.criterion(scores.log(), target)

        # Compute sum of perplexities for stats
        loss_data = loss.sum().data.clone()
        stats = self._stats(loss_data, scores.data, target.data)

        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            # Compute Sequence Lengths
            pad_ix = batch.dataset.fields['tgt'].vocab.stoi[inputters.PAD_WORD]
            tgt_lens = batch.tgt.ne(pad_ix).float().sum(0)
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats

    def monolithic_compute_loss(self, batch, output, attns):
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns)
        source = inputters.make_features(batch, 'src', 'text')  # shouldn't be sharded
        _, batch_stats = self._compute_loss(batch, source=source, **shard_state)
        return batch_stats

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization):
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns)
        source = inputters.make_features(batch, 'src', 'text')  # shouldn't be sharded

        batch_stats = Statistics()
        for shard in loss.shards(shard_state, shard_size):
            l, stats = self._compute_loss(batch, source=source, **shard)
            l.div(normalization).backward()
            batch_stats.update(stats)

        return batch_stats
