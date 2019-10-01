import json
from typing import Sequence, List, Dict, NamedTuple

import six
from torchtext.data import RawField

from onmt.inputters.datareader_base import DataReaderBase


class PrefixTreeNode(NamedTuple):
    word: int
    children: dict

    def add(self, token_stack: List[int], termination_children: Dict[int, 'PrefixTreeNode']):
        if len(token_stack) == 0:
            self.children.update(termination_children)
            return

        top = token_stack.pop()
        if top not in self.children:
            self.children[top] = PrefixTreeNode(top, {})
        self.children[top].add(token_stack, termination_children)


class MentionDataReader(DataReaderBase):
    def read(self, sequences, side, _dir=None):
        """Read text data from disk.

        Args:
            sequences (str or Iterable[str]):
                path to text file or iterable of the actual text data.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with MentionDataReader."
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)
        for i, seq in enumerate(sequences):
            if isinstance(seq, six.binary_type):
                seq = seq.decode("utf-8")
            yield {side: json.loads(seq), "indices": i}


class PrefixTreeField(RawField):
    def __init__(self, vocab):
        super(PrefixTreeField, self).__init__(preprocessing=self.build_prefix_tree)
        self.vocab = vocab
        self.sep = vocab.stoi['<SEP>']
        self.bos = vocab.stoi['<s>']
        self.eos = vocab.stoi['</s>']
        self.relation_symbols = [i for i, s in enumerate(self.vocab.itos)
                                 if s.startswith('<P') and s.endswith('>')]

    def build_prefix_tree(self, mentions: Sequence[List[str]]) -> PrefixTreeNode:
        sep_node = PrefixTreeNode(self.sep, {})
        eos_node = PrefixTreeNode(self.eos, {})
        eos_node.children[self.eos] = eos_node  # loop infinitely after sentence termination

        termination_children = {self.sep: sep_node, self.eos: eos_node}
        for mention in mentions:
            stack = [self.vocab.stoi[token] for token in reversed(mention)]
            sep_node.add(stack, termination_children)

        root_node = PrefixTreeNode(self.bos, {})
        for r in self.relation_symbols:
            root_node.children[r] = PrefixTreeNode(r, sep_node.children)
        return root_node
