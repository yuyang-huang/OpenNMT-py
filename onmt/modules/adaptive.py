import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss

from onmt.utils.loss import LossComputeBase
from onmt.modules.shared_vocab_copy_generator import collapse_copy_scores


class AdaptiveCopyGenerator(AdaptiveLogSoftmaxWithLoss):
    def __init__(self, in_features, n_classes, cutoffs,
                 div_value=4., head_bias=False, adaptive_input=None):
        if adaptive_input is not None:
            embedding_size = adaptive_input.embedding_size
        else:
            embedding_size = in_features
        super(AdaptiveCopyGenerator, self).__init__(embedding_size, n_classes, cutoffs,
                                                    div_value=div_value,
                                                    head_bias=head_bias)
        if in_features != self.in_features:
            self.linear_in = nn.Linear(in_features, self.in_features)
        else:
            self.linear_in = None

        self.linear_copy = nn.Linear(self.in_features, 1)

        self.tied_embeddings = adaptive_input is not None
        if self.tied_embeddings:
            if adaptive_input.cutoffs != self.cutoffs:
                raise ValueError('To tie weights between adaptive input/output, '
                                 'their cutoffs must be identical. '
                                 f'Got input={adaptive_input.cutoffs} output={self.cutoffs}')
            self.head = HeadLinear(self.in_features, self.head_size, adaptive_input.head[0],
                                   bias=head_bias)
            for output_tail, input_tail in zip(self.tail, adaptive_input.tail):
                output_tail[-1].weight = input_tail[0].weight

        self.reset_parameters()

    def forward(self, hidden, attn, src, batch_first=False):
        if self.linear_in is not None:
            hidden = self.linear_in(hidden)

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


class AdaptiveInput(nn.Module):
    def __init__(self, embeddings_size, n_classes, cutoffs,
                 div_value=4., head_bias=False, padding_idx=None):
        super(AdaptiveInput, self).__init__()

        cutoffs = list(cutoffs)
        if (cutoffs != sorted(cutoffs)
                or min(cutoffs) <= 0
                or max(cutoffs) >= (n_classes - 1)
                or len(set(cutoffs)) != len(cutoffs)
                or any(int(c) != c for c in cutoffs)):
            raise ValueError("cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and n_classes-1")

        self.embedding_size = embeddings_size
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias

        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.cutoffs[0]

        self.head = nn.Sequential(nn.Embedding(self.head_size, self.embedding_size,
                                               padding_idx=padding_idx),
                                  nn.Linear(self.embedding_size, self.embedding_size,
                                            bias=self.head_bias))
        self.tail = nn.ModuleList()

        for i in range(self.n_clusters):
            hsz = int(self.embedding_size // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = nn.Sequential(
                nn.Embedding(osz, hsz),
                nn.Linear(hsz, self.embedding_size, bias=False),
            )

            self.tail.append(projection)

    def forward(self, input):
        output_size = (*input.size()[:-1], self.embedding_size)
        input = input.view(-1)

        used_rows = 0
        expected_rows = input.size(0)
        output = input.new_zeros((expected_rows, self.embedding_size)).float()

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            input_mask = (input >= low_idx) & (input < high_idx)
            row_indices = input_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue

            if i == 0:
                out = self.head(input[input_mask] - low_idx)
            else:
                out = self.tail[i - 1](input[input_mask] - low_idx)

            output.index_copy_(0, row_indices, out)
            used_rows += row_indices.numel()

        if used_rows != expected_rows:
            raise RuntimeError("Target values should be in [0, {}], "
                               "but values in range [{}, {}] "
                               "were found. ".format(self.n_classes - 1,
                                                     input.min().item(),
                                                     input.max().item()))
        return output.view(*output_size)

    def update_dropout(self, p):
        raise NotImplementedError

    def load_pretrained_vectors(self, path):
        assert not path


class HeadLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tied_embedding: nn.Embedding,
                 bias: bool = False):
        super(HeadLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_clusters = out_features - tied_embedding.weight.size(0)
        self.tied_weight = tied_embedding.weight
        self.weight = nn.Parameter(torch.Tensor(self.n_clusters, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = torch.cat((self.tied_weight, self.weight), 0)
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, n_clusters={}, tied_embedding.size={}'.format(
            self.in_features, self.out_features, self.n_clusters, self.tied_weight.size())
