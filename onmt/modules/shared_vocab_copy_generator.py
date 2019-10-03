""" Generator module """
import torch.nn as nn
import torch
import torch.cuda

from onmt.utils.loss import LossComputeBase


def collapse_copy_scores(prob, p_copy, attn, src, batch_first=False):
    """Merge softmax distribution with copy attention.

    Parameters
    ----------
    prob : torch.FloatTensor
        The softmax distribution, size is ``[tgt_len * batch_size, vocab_size]``.
    p_copy : torch.FloatTensor
        The Bernoulli variable indicating whether to copy, size is ``[tgt_len * batch_size, 1]``.
    attn : torch.FloatTensor
        The attention alignment, size is ``[tgt_len * batch_size, src_len]``.
    src : torch.LongTensor
        The integer indices of source words of size ``[slen, batch, 1]``.
    batch_first : bool
        If True, then the 0th dimension of the inputs will be interpreted as
        ``batch_size * tgt_len`` instead of ``tgt_len * batch_size``.

    Returns
    -------
    log_prob : torch.FloatTensor
        The final log-probability distribution, of size ``[tgt_len * batch_size, vocab_size]``
        (or ``[batch_size * tgt_len, vocab_size]`` if `batch_first` is True.
    """
    # weighted by p(z=0) and reshape to [tlen, batch_size * vocab_size] for index_add
    batch_size = src.size(1)
    tgt_length = prob.size(0) // batch_size
    prob = prob.mul(1 - p_copy)
    if batch_first:
        prob = prob.view(batch_size, tgt_length, -1).transpose(0, 1).contiguous()

    # gather attention scores on src tokens and add to the output distribution
    vocab_size = prob.size(-1)
    batch_offset = torch.arange(0, batch_size * vocab_size,
                                step=vocab_size,
                                device=src.device).unsqueeze(1)
    src = src.transpose(0, 1).squeeze(-1).add(batch_offset).contiguous().view(-1)

    attn = attn.mul(p_copy)
    if batch_first:
        attn = attn.view(batch_size, tgt_length, -1).transpose(0, 1).contiguous()

    prob = prob.view(tgt_length, -1).index_add(-1, src, attn.view(tgt_length, -1))
    if batch_first:
        prob = prob.view(tgt_length, batch_size, vocab_size).transpose(0, 1).contiguous()

    # clamp just in case numerical error occurs
    return torch.clamp(prob.view(-1, vocab_size), 1e-20).log()


class SharedVocabCopyGenerator(nn.Module):
    """Copy generator with shared vocabulary (faster implementation).
    """
    def __init__(self, input_size, output_size, pad_idx,
                 generator_linear=None):
        self.pad_idx = pad_idx
        super().__init__()
        if generator_linear is None:
            self.linear = nn.Linear(input_size, output_size)
        else:
            self.linear = generator_linear
        self.linear_copy = nn.Linear(input_size, 1)

    def forward(self, hidden, attn, src, batch_first=False):
        # probability of copying p(z=1)
        p_copy = torch.sigmoid(self.linear_copy(hidden))

        # original distribution
        prob = torch.softmax(self.linear(hidden), 1)

        return collapse_copy_scores(prob, p_copy, attn, src, batch_first=batch_first)


class SharedVocabCopyGeneratorLossCompute(LossComputeBase):
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
