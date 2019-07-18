import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.utils.misc import aeq, sequence_mask


class NGramAttention(nn.Module):
    def __init__(self, query_dim, memory_dim, max_order,
                 attend_to='encoder', merge_mode='pooling'):
        super(NGramAttention, self).__init__()
        self.max_order = max_order
        self.attend_to = attend_to
        self.merge_mode = merge_mode

        # use order=0 for global attention, order>0 for n-gram attentions
        linear_in = [nn.Linear(query_dim, query_dim, bias=False)]
        linear_out = [nn.Linear(query_dim * 3, query_dim, bias=False)]
        for order in range(1, max_order + 1):
            if self.merge_mode == 'concat':
                latent_dim = memory_dim * order
            else:
                # mean-max pooling is the concatenation of two tensors
                latent_dim = memory_dim * 2
            linear_in.append(nn.Linear(query_dim, latent_dim, bias=False))
            linear_out.append(nn.Linear(latent_dim, query_dim, bias=False))

        self.linear_in = nn.ModuleList(linear_in)
        self.linear_out = nn.ModuleList(linear_out)

    def score(self, query, memory_bank, order=0):
        batch, query_len, query_dim = query.size()
        batch_, memory_len, memory_dim = memory_bank.size()
        aeq(batch, batch_)
        aeq(self.linear_in[order].in_features, query_dim)
        aeq(self.linear_in[order].out_features, memory_dim)

        query_ = query.view(batch * query_len, query_dim)
        query_ = self.linear_in[order](query_)
        query = query_.view(batch, query_len, memory_dim)

        # (batch, query_len, d) x (batch, d, memory_len) --> (batch, query_len, memory_len)
        return torch.bmm(query, memory_bank.transpose(1, 2))

    def forward(self, query, memory_bank, memory_lengths=None):
        # one step input
        if query.dim() == 2:
            one_step = True
            query = query.unsqueeze(1)
        else:
            one_step = False

        # split memory bank into encoder outputs and word embeddings
        memory_bank, emb = memory_bank
        if self.attend_to == 'encoder':
            emb = memory_bank

        # compute context vectors
        global_context, global_alignment = self.compute_global_context(
            query, memory_bank, memory_lengths=memory_lengths)
        ngram_context, ngram_alignment = self.compute_ngram_context(
            query, emb, memory_lengths=memory_lengths)

        # concatenate
        batch, query_len, query_dim = query.size()
        concat_context = torch.cat([query, global_context, ngram_context], 2)
        concat_context = concat_context.view(batch * query_len, query_dim * 3)
        attn_h = self.linear_out[0](concat_context).view(batch, query_len, query_dim)

        attn_h, alignments = self.reshape_output(
            attn_h, global_alignment, ngram_alignment, one_step)
        self.verify_output_shape(attn_h, alignments, query, memory_bank, one_step)
        return attn_h, alignments

    def compute_global_context(self, query, memory_bank, memory_lengths=None):
        batch, query_len, query_dim = query.size()
        batch_, memory_len, memory_dim = memory_bank.size()

        align = self.score(query, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))

        align_vectors = F.softmax(align.view(batch * query_len, memory_len), -1)
        align_vectors = align_vectors.view(batch, query_len, memory_len)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        return torch.bmm(align_vectors, memory_bank), align_vectors

    def compute_ngram_context(self, query, memory_bank, memory_lengths=None):
        batch, query_len, query_dim = query.size()
        batch_, memory_len, memory_dim = memory_bank.size()

        # generate n-gram contexts
        ngram_contexts = []
        ngram_alignments = []

        for order in range(1, self.max_order + 1):
            num_ngrams = memory_len - order + 1
            if num_ngrams <= 0:
                continue

            # shift and stack token embeddings
            token_emb = []
            for offset in range(order):
                token_emb.append(memory_bank[:, offset:(offset + num_ngrams), :])

            # merge the token embeddings to form the n-gram embedding
            if self.merge_mode == 'concat':
                ngram_emb = torch.cat(token_emb, dim=-1)
            else:
                # mean-max pooling
                stacked_emb = torch.cat([x.unsqueeze(0) for x in token_emb], dim=0)
                ngram_emb = torch.cat([stacked_emb.mean(dim=0),
                                       stacked_emb.max(dim=0).values], dim=-1)

            align = self.score(query, ngram_emb, order=order)
            if memory_lengths is not None:
                mask = sequence_mask(memory_lengths - order + 1, max_len=align.size(-1))

                # XXX: workaround for sequences with length less than `order`,
                # where the output of ``F.softmax()`` will be NaN.
                # unmask a random element so that `align` won't be full of ``-inf``
                has_any_nonzero = mask.any(dim=-1)
                mask[~has_any_nonzero, 0] = 1

                mask = mask.unsqueeze(1)  # Make it broadcastable.
                align.masked_fill_(1 - mask, -float('inf'))

            align_vectors = F.softmax(align.view(batch * query_len, num_ngrams), -1)
            align_vectors = align_vectors.view(batch, query_len, num_ngrams)

            if memory_lengths is not None:
                # XXX: we've randomly unmasked the first element
                # now multiply it by zero to prevent it from affecting the loss
                mask = memory_lengths.ge(order).view(batch, 1, 1).float()
                align_vectors = align_vectors.mul(mask)
            ngram_alignments.append(align_vectors)

            c = torch.bmm(align_vectors, ngram_emb)
            ngram_contexts.append(self.linear_out[order](c))

        return torch.tanh(sum(ngram_contexts)), ngram_alignments

    def reshape_output(self, attn_h, global_alignment, ngram_alignment, one_step):
        if one_step:
            attn_h = attn_h.squeeze(1)
            alignments = {
                'std': global_alignment.squeeze(1),
                'ngram': [x.squeeze(1) for x in ngram_alignment],
            }
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            alignments = {
                'std': global_alignment.transpose(0, 1).contiguous(),
                'ngram': [x.transpose(0, 1).contiguous() for x in ngram_alignment],
            }
        return attn_h, alignments

    def verify_output_shape(self, attn_h, alignments, query, memory_bank, one_step):
        batch, query_len, query_dim = query.size()
        memory_len = memory_bank.size(1)

        if one_step:
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(query_dim, dim_)
            batch_, memory_len_ = alignments['std'].size()
            aeq(batch, batch_)
            aeq(memory_len, memory_len_)
            for order in range(len(alignments['ngram'])):
                batch_, memory_len_ = alignments['ngram'][order].size()
                aeq(batch, batch_)
                aeq(memory_len - order, memory_len_)
        else:
            query_len_, batch_, query_dim_ = attn_h.size()
            aeq(query_len, query_len_)
            aeq(batch, batch_)
            aeq(query_dim, query_dim_)
            query_len_, batch_, memory_len_ = alignments['std'].size()
            aeq(query_len, query_len_)
            aeq(batch, batch_)
            aeq(memory_len, memory_len_)
            for order in range(len(alignments['ngram'])):
                query_len_, batch_, memory_len_ = alignments['ngram'][order].size()
                aeq(query_len, query_len_)
                aeq(batch, batch_)
                aeq(memory_len - order, memory_len_)

    def extra_repr(self):
        args = []
        for attr in ('max_order', 'attend_to', 'merge_mode'):
            value = str(getattr(self, attr))
            args.append('='.join((attr, value)))
        return ', '.join(args)
