"""
 RNN tools
"""
import torch.nn as nn
import onmt.models
from onmt.modules import WeightDroppedLSTM


def rnn_factory(rnn_type, num_layers, dropout, dropout_type, **kwargs):
    """ rnn factory, Use pytorch version when available. """
    no_pack_padded_seq = rnn_type == 'SRU' or dropout > 0
    if rnn_type == "SRU":
        # SRU doesn't support PackedSequence.
        rnn = onmt.models.sru.SRU(num_layers=num_layers,
                                  dropout=dropout,
                                  **kwargs)
    else:
        if dropout_type == 'variational':
            assert rnn_type == 'LSTM'
            rnn_cls = WeightDroppedLSTM
        else:
            rnn_cls = getattr(nn, rnn_type)
        rnn = nn.ModuleList([rnn_cls(**kwargs) for _ in range(num_layers)])
    return rnn, no_pack_padded_seq
