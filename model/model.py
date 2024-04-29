from model.modules import *
import torch
import torch.nn as nn


class RNAModel(nn.Module):

    def __init__(self, matrix_shape:int = 512):
        
        super(RNAModel, self).__init__()
        self.blocks = []
        for i in range(5):
            self.blocks.append(ResNetBlock(in_channels=2**i,out_channels=2**(i+1),kernel_size=3))
            matrix_shape = int(matrix_shape/2)

        self.blocks.append(FCLayer(in_channels=2**(i+1),shape_x=matrix_shape,shape_y=matrix_shape))

    def forward(self,x):
        for block in self.blocks:
            x = block(x)
        return x
