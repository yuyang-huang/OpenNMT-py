import torch.nn as nn
import torch.nn.functional as F


class VariationalDropout(nn.Module):
    def __init__(self, p=0.0):
        super(VariationalDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        ones = x.new_ones(1, x.size(1), x.size(-1))
        mask = F.dropout(ones, p=self.p, training=self.training)
        return x * mask

    def extra_repr(self):
        return 'p={}'.format(self.p)
