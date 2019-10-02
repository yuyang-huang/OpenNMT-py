""" Generator module """
import torch.nn as nn
import torch
import torch.cuda
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss

from onmt.utils.loss import LossComputeBase
from onmt.modules.shared_vocab_copy_generator import collapse_copy_scores


class AdaptiveCopyGenerator(AdaptiveLogSoftmaxWithLoss):
    def __init__(self, in_features, n_classes, cutoffs,
                 div_value=4., head_bias=False):
        super(AdaptiveCopyGenerator, self).__init__(in_features, n_classes, cutoffs,
                                                    div_value=div_value,
                                                    head_bias=head_bias)
        self.linear_copy = nn.Linear(in_features, 1)

    def forward(self, hidden, attn, src, batch_first=False):
        # probability of copying p(z=1)
        p_copy = torch.sigmoid(self.linear_copy(hidden))

        # Adaptive Softmax distribution
        prob = super().log_prob(hidden).exp()

        return collapse_copy_scores(prob, p_copy, attn, src, batch_first=batch_first)


class AdaptiveCopyGeneratorLossCompute(LossComputeBase):
    def __init__(self, criterion, generator, normalize_by_length):
        super().__init__(criterion, generator)
        self.normalize_by_length = normalize_by_length

    def _make_shard_state(self, batch, output, range_, attns):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1], :, 0],
            "copy_attn": attns["copy"],
        }

    def _compute_loss(self, batch, output, target, copy_attn):
        """Compute the loss. The args must match self._make_shard_state().

        Parameters
        ----------
        batch : torchtext.data.batch.Batch
            The current batch.
        output : torch.FloatTensor
            Hidden outputs from the decoder, of size ``[tlen, batch, input_size]``.
        target : torch.LongTensor
            The integer indices of source words, of size ``[tlen, batch]``.
        copy_attn : torch.FloatTensor
            Attention over the source tokens, of size ``[tlen, batch, slen]``.

        Returns
        -------
        loss : torch.FloatTensor
            The scalar loss value.
        stats : onmt.utils.Statistics
            The statistics of this batch.
        """
        target = target.view(-1)
        scores = self.generator(self._bottle(output),
                                self._bottle(copy_attn),
                                batch.src[0])
        loss = self.criterion(scores, target)

        # Compute sum of perplexities for stats
        loss_data = loss.sum().clone()
        stats = self._stats(loss_data, scores, target)

        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            # Compute Sequence Lengths
            tgt_lens = batch.tgt[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats
