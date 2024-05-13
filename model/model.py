from model.modules import *
import torch
import torch.nn as nn


class RNAModel(nn.Module):

    def __init__(self, batch_size, matrix_shape: int = 512):
        
        super(RNAModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = DecoderLSTM(batch_size)
        self.batch_size = batch_size

    def forward(self, matrices, sequences):
        encoded = self.encoder(matrices)
        encoded = torch.reshape(encoded, (1, self.batch_size, 128))
        x = self.decoder(sequences, encoded)
        return x


