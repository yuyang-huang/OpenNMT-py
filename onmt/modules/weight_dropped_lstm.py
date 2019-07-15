import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightDroppedLSTM(nn.LSTM):
    """Apply DropConnect to the hidden-to-hidden connections of the LSTM.

    Ref. Merity et al., ICLR'18, Regularizing and optimizing LSTM language models.
    """
    def forward_impl(self, input, hx, batch_sizes, max_batch_size, sorted_indices):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor], int, Optional[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        # DropConnect: randomly drop some of the recurrent connections
        weights = []
        for weight_name, weight_tensor in zip(self._get_flat_weights_names(),
                                              self._get_flat_weights()):
            if weight_name.startswith('weight_hh'):
                weight_tensor = F.dropout(weight_tensor,
                                          p=self.dropout,
                                          training=self.training)
            weights.append(weight_tensor)

        self.check_forward_args(input, hx, batch_sizes)

        # set built-in dropout probability to 0.0 to disable it
        if batch_sizes is None:
            result = nn._VF.lstm(input, hx, weights, self.bias, self.num_layers,
                                 0.0, self.training, self.bidirectional, self.batch_first)
        else:
            result = nn._VF.lstm(input, batch_sizes, hx, weights, self.bias,
                                 self.num_layers, 0.0, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]

        return output, hidden
