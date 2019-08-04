import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.modules import Embeddings


class StructuredJointEmbeddings(nn.Module):
    """The structure-aware joint input/output embeddings output layer.

    Ref: Beyond Weight Tying: Learning Joint Input-Output Embeddings for Neural Machine Translation
    (Pappas et al., WMT'18)
    """
    def __init__(self, in_features: int, latent_features: int, out_features: int,
                 embeddings: Embeddings):
        super(StructuredJointEmbeddings, self).__init__()
        self.in_features = in_features
        self.latent_features = latent_features
        self.out_features = out_features

        self.weight_output_structure = nn.Parameter(torch.Tensor(latent_features,
                                                                 embeddings.word_vec_size))
        self.bias_output_structure = nn.Parameter(torch.Tensor(latent_features))

        self.weight_hidden_structure = nn.Parameter(torch.Tensor(latent_features, in_features))
        self.bias_hidden_structure = nn.Parameter(torch.Tensor(latent_features))

        self.bias = nn.Parameter(torch.Tensor(out_features))  # always use bias in output layer
        self.embedding_weight = embeddings.word_lut.weight  # tied input/output weight
        self.reset_parameters()

    def reset_parameters(self):
        """Avoid re-initializing the embedding matrix.
        """
        nn.init.kaiming_uniform_(self.weight_output_structure, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_hidden_structure, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.embedding_weight.size(1))
        nn.init.uniform_(self.bias_output_structure, -bound, bound)
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias_hidden_structure, -bound, bound)
        bound = 1 / math.sqrt(self.latent_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output_structure = F.linear(self.embedding_weight,
                                    self.weight_output_structure,
                                    self.bias_output_structure)  # [vocab_size, latent_dim]
        hidden_structure = F.linear(input,
                                    self.weight_hidden_structure,
                                    self.bias_hidden_structure)  # [*, latent_dim]
        return F.linear(hidden_structure, output_structure, self.bias)  # [*, vocab_size]

    def extra_repr(self):
        return 'in_features={}, latent_features={}, out_features={}'.format(
            self.in_features, self.latent_features, self.out_features)
