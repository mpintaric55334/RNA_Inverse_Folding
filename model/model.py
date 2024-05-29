from model.modules import *
import torch
import torch.nn as nn


class RNAModel(nn.Module):

    def __init__(self, batch_size, matrix_shape: int = 512, hidden_size=256,
                 encoder_channels=128):
        
        super(RNAModel, self).__init__()
        self.encoder = Encoder(in_channels=1, desired_channels=encoder_channels,
                               matrix_shape=matrix_shape)
        self.decoder = DecoderLSTM(hidden_size=hidden_size)
        self.batch_size = batch_size
        self.states = (torch.zeros(1, batch_size, hidden_size),
                       torch.zeros(1, batch_size, hidden_size))
        self.infer_first_iter = False
        self.hidden_size = hidden_size

    def forward(self, matrices, sequences):
        device = sequences.device
        encoded = self.encoder(matrices)
        encoded = torch.reshape(encoded, (1, self.batch_size, self.hidden_size))
        self.states = (self.states[0].to(device), encoded.to(device))  # give encoding
        x, _ = self.decoder(sequences, self.states)
        return x
    
    def infer(self, matrices, sequences):
        device = sequences.device
        encoded = self.encoder(matrices)
        encoded = torch.reshape(encoded, (1, self.batch_size, self.hidden_size))
        if not self.infer_first_iter:
            self.states = (self.states[0].to(device), encoded.to(device))  # give encoding
            self.infer_first_iter = True
        x, self.states = self.decoder(sequences, self.states)
        return x
    
    def reset_states(self):
        self.states = (torch.zeros(1, self.batch_size, self.hidden_size),
                       torch.zeros(1, self.batch_size, self.hidden_size))
        self.infer_first_iter = False

    def reset_batch_size(self, new_batch_size):
        self.batch_size = new_batch_size
