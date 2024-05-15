from model.modules import *
import torch
import torch.nn as nn


class RNAModel(nn.Module):

    def __init__(self, batch_size, matrix_shape: int = 512):
        
        super(RNAModel, self).__init__()
        self.encoder = Encoder(1, 128, matrix_shape)
        self.decoder = DecoderLSTM(batch_size)
        self.batch_size = batch_size
        self.states = (torch.zeros(1, batch_size, 256),
                       torch.zeros(1, batch_size, 256))
        self.infer_first_iter = False

    def forward(self, matrices, sequences):
        device = sequences.device
        encoded = self.encoder(matrices)
        encoded = torch.reshape(encoded, (1, self.batch_size, 256))
        self.states = (self.states[0].to(device), encoded.to(device))  # give encoding
        x, _ = self.decoder(sequences, self.states)
        return x
    
    def infer(self, matrices, sequences):
        device = sequences.device
        encoded = self.encoder(matrices)
        encoded = torch.reshape(encoded, (1, self.batch_size, 256))
        if not self.infer_first_iter:
            self.states = (self.states[0].to(device), encoded.to(device))  # give encoding
            self.infer_first_iter = True
        x, self.states = self.decoder(sequences, self.states)
        return x
    
    def reset_states(self):
        self.states = (torch.zeros(1, self.batch_size, 256),
                       torch.zeros(1, self.batch_size, 256))
        self.infer_first_iter = False
