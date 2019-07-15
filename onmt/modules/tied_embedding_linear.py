import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from onmt.modules import Embeddings


class TiedEmbeddingLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, embeddings: Embeddings):
        super(TiedEmbeddingLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(embeddings.word_vec_size, in_features))
        self.bias = Parameter(torch.Tensor(out_features))  # always use bias in output layer
        self.embedding_weight = embeddings.word_lut.weight  # tied input/output weight
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = torch.tanh(self.embedding_weight.matmul(self.weight))
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, embedding.size={}'.format(
            self.in_features, self.out_features, self.embedding_weight.size())
