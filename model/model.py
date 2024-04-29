from model.modules import *
import torch
import torch.nn as nn


class RNAModel(nn.Module):

    def __init__(self, matrix_shape:int = 512):
        
        super(RNAModel, self).__init__()
        self.encoder = Encoder()

    def forward(self,x):
        x = self.encoder(x)
        return x
