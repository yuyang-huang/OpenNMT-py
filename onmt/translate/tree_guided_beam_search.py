import torch

from onmt.translate.beam_search import BeamSearch


class TreeGuidedBeamSearch(BeamSearch):
    def __init__(self, prefix_trees, sep, **kwargs):
        super(TreeGuidedBeamSearch, self).__init__(**kwargs)

        # each beam starts from the root node of the corresponding example
        self.sep = sep
        self.prefix_tree_ptrs = [[prefix_trees[b] for _ in range(self.beam_size)]
                                 for b in range(self.batch_size)]

        # disable some unused functionalities
        assert not self.global_scorer.has_len_pen
        assert not self.global_scorer.has_cov_pen
        assert not self.return_attention
        assert self.block_ngram_repeat == 0

    def topk(self, candidate_scores):
        vocab_size = candidate_scores.size(-1)
        _B = candidate_scores.shape[0] // self.beam_size
        candidate_scores = candidate_scores.reshape(_B, self.beam_size * vocab_size)

        torch.topk(candidate_scores, self.beam_size, dim=-1,
                   out=(self.topk_scores, self.topk_ids))

        # resolve beam origin and map to batch index flat representation
        torch.div(self.topk_ids, vocab_size, out=self._batch_index)
        self.topk_ids.fmod_(vocab_size)  # resolve true word ids

    def advance(self, log_probs):
        _B = log_probs.shape[0] // self.beam_size

        # force the output to be longer than self.min_length
        self.ensure_min_length(log_probs)

        # multiply next word probs by the beam probability
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        # prune invalid candidates according to the prefix trees and take top-K
        log_probs = self.apply_prefix_tree_constraints(log_probs)
        self.topk(log_probs)
        self.advance_prefix_tree_ptrs()

        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)

        # append last prediction
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(_B * self.beam_size, 1)], -1)

        self.is_finished = self.topk_ids.eq(self.eos)
        self.ensure_max_length()

    def update_finished(self):
        non_finished = super().update_finished()
        self.prefix_tree_ptrs = [self.prefix_tree_ptrs[b] for b in non_finished]

    def apply_prefix_tree_constraints(self, log_probs):
        # Only the elements matching the `True` positions in `mask` will be
        # kept. All other elements will be zeroed out.
        mask = torch.zeros_like(log_probs, dtype=torch.bool)

        nodes = (node for nodes in self.prefix_tree_ptrs for node in nodes)
        for i, node in enumerate(nodes):
            if self.eos in node.children:
                # end of a mention: either terminate the sentence (</s>) or
                # decode the tail mention (<SEP>)
                has_generated_sep_token = self.alive_seq[i].eq(self.sep).any()
                if has_generated_sep_token:
                    valid_next_token_indices = [x for x in node.children if x != self.sep]
                else:
                    valid_next_token_indices = [x for x in node.children if x != self.eos]
            else:
                valid_next_token_indices = list(node.children)

            mask[i, valid_next_token_indices] = True

        return log_probs.where(mask, torch.tensor(-1e20, device=mask.device))

    def advance_prefix_tree_ptrs(self):
        next_prefix_tree_ptrs = []

        for batch_idx in range(len(self._batch_index)):
            beam_origins = self._batch_index[batch_idx]
            next_word_indices = self.topk_ids[batch_idx]

            batch_prefix_tree_ptrs = []
            for beam_idx, word_idx in zip(beam_origins, next_word_indices):
                # traverse the prefix tree to get the next node
                node = self.prefix_tree_ptrs[batch_idx][beam_idx]
                next_node = node.children.get(word_idx.item(), node)
                batch_prefix_tree_ptrs.append(next_node)

            next_prefix_tree_ptrs.append(batch_prefix_tree_ptrs)

        self.prefix_tree_ptrs = next_prefix_tree_ptrs
