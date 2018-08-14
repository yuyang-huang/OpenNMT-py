import codecs

import torch
import torchtext

from onmt.inputters.dataset_base import (
    DatasetBase,
    PAD_WORD,
    BOS_WORD,
    EOS_WORD,
)
from onmt.inputters.text_dataset import ShardedTextCorpusIterator


class ClusterDataset(DatasetBase):
    """ Dataset for data_type=='cluster'

        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
                Keys are like 'src', 'tgt', 'cluster'.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            src_seq_length (int): maximum source sequence length.
            tgt_seq_length (int): maximum target sequence length.
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """

    def __init__(self, fields, src_examples_iter, tgt_examples_iter, cluster_examples_iter,
                 src_seq_length=0, tgt_seq_length=0, use_filter_pred=True, ignore_noise=False):
        # Attributes used elsewhere.
        self.data_type = 'cluster'
        self.n_src_feats = 0
        self.n_tgt_feats = 0
        self.src_vocabs = []

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        if tgt_examples_iter is not None:
            zipped_iter = zip(src_examples_iter, tgt_examples_iter, cluster_examples_iter)
        else:
            zipped_iter = zip(src_examples_iter, cluster_examples_iter)
        examples_iter = (self._join_dicts(*zipped) for zipped in zipped_iter)

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        src_size = 0

        out_examples = []
        for ex_values in example_values:
            example = self._construct_example_fromlist(ex_values, out_fields)
            src_size += len(example.src)
            out_examples.append(example)

        def filter_pred(example):
            return (
                0 < len(example.src) <= src_seq_length
                and 0 < len(example.tgt) <= tgt_seq_length
                and (not ignore_noise or example.cluster >= 0)
            )

        filter_pred = filter_pred if use_filter_pred else lambda x: True
        super(ClusterDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        # Default to a balanced sort, prioritizing tgt len match.
        if hasattr(ex, "tgt"):
            return len(ex.src), len(ex.tgt)
        return len(ex.src)

    @staticmethod
    def make_text_examples_nfeats_tpl(cluster_path):
        text_iter = ClusterDataset.make_text_iterator_from_file(cluster_path)
        examples_iter = ClusterDataset.make_examples(text_iter)
        return examples_iter

    @staticmethod
    def make_examples(text_iter):
        for i, line in enumerate(text_iter):
            cluster, prob = line.split()
            yield {'cluster': int(cluster), 'prob': float(prob), 'indices': i}

    @staticmethod
    def make_text_iterator_from_file(path):
        with codecs.open(path, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                yield line

    @staticmethod
    def get_fields(n_src_features, n_tgt_features):
        """
        Args:
            n_src_features (int): the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features (int): the number of target features to
                create `torchtext.data.Field` for.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

        for j in range(n_src_features):
            fields["src_feat_" + str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        for j in range(n_tgt_features):
            fields["tgt_feat_" + str(j)] = \
                torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,
                                     pad_token=PAD_WORD)

        fields["cluster"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False)

        fields["prob"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False)

        return fields

    @staticmethod
    def get_batch_size_fn(max_batch_size):
        """
        Arguments
        ---------
        max_batch_size : int
            Will try to put all examples of the same cluster in the same batch,
            but to prevent OOM, at most `max_batch_size` examples will be
            generated. Users are expected to take care of post-processing the
            predictions.
        """
        current_cluster = None

        def batch_size_fn(new, count, sofar):
            """
            Arguments
            ---------
            new : Example
                The latest example added to the list `mini_batch`.
            count : int
                The length of the list `mini_batch`.
            sofar : int
                Accumulated value returned by this function so far.
                Will be reset to zero once `sofar` reaches `opt.batch_size`,
                and yield the mini-batch (`mini_batch` will also be cleared).
            """
            nonlocal current_cluster
            if current_cluster is None:
                # first batch
                current_cluster = new.cluster

            # samples are assumed to be sorted according to cluster index
            if count == max_batch_size or new.cluster != current_cluster:
                # emit the batch
                current_cluster = new.cluster
                return max_batch_size + 1
            else:
                # still within the same cluster
                return 0
        return batch_size_fn


class ShardedClusterIterator(ShardedTextCorpusIterator):
    def _example_dict_iter(self, line, index):
        cluster, prob = line.split()
        example_dict = {'cluster': int(cluster), 'prob': float(prob), 'indices': index}
        return example_dict
