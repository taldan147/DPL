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
        return h_t[self.num_of_layers-1].view(-1, 1, self.hs_size)

class Decoder(nn.Module):
    def __init__(self, input_size, num_of_layers, seq_size, hs_size, dropout, output_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.num_of_layers = num_of_layers
        self.hs_size = hs_size
        self.dropout = dropout
        self.seq_size = seq_size
        self.class_size = 10
        self.linear = nn.Linear(hs_size, output_size)
        self.linearClass = nn.Linear(hs_size, self.class_size)
        self.lstmDecoder = nn.LSTM(hs_size, hs_size, batch_first=True, dropout=dropout)

    def forward(self, x: torch.tensor):
        x = x.repeat(1, self.seq_size, 1)
        output, (hidden, _) = self.lstmDecoder(x)
        return self.linear(output), self.linearClass(hidden)

class LSTMAE(nn.Module):
    def __init__(self, input_size, num_of_layers, seq_size, hs_size, dropout, output_size):
        super(LSTMAE, self).__init__()
        self.input_size = input_size
        self.num_of_layers = num_of_layers
        self.hs_size = hs_size
        self.dropout = dropout
        self.encoder = Encoder(input_size, num_of_layers, hs_size, dropout)
        self.decoder = Decoder(input_size, num_of_layers, seq_size, hs_size, dropout, output_size)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

