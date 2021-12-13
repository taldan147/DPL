import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, num_of_layers, hs_size, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.num_of_layers = num_of_layers
        self.hs_size = hs_size
        self.dropout = dropout
        self.lstmEncoder = nn.LSTM(input_size, hs_size, num_of_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        output, (h_t, c_t) = self.lstmEncoder(x)
