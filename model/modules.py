import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class ResNetLayer(nn.Module):

    """
    Basic building block of the resnet model. Consists of two convolution layers, each layer having same dilation,
    and same kernel size. After each layer, batchnorm and relu are done.

    Arguments:
        kernel_size: size of the kernel
        in_channels: number of in channels
        out_channels: number of out channels
        dilation: dilation number
    """

    def __init__(self, kernel_size: int, in_channels: int, out_channels: int,dilation: int):

        super(ResNetLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size,padding="same",dilation=dilation)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size,padding="same",dilation=dilation)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self,x):

        x_input = x
        x = self.relu1(self.batchnorm1(self.conv1(x)))
        x = self.relu2(self.batchnorm2(self.conv2(x)))
        return x

class ResNetBlock(nn.Module):

    """
    Larger block of the module, consists of three resnet layers, so all 3 dilations are used in one block.
    At the beginning of the block, input goes through prijection layer, so the residual connection has the same 
    number of channels as conv output, enabling addition. Each resnet layer has same number of in and out channels as 
    the others in the block. At the end of the block, max pooling is done.

    Arguments:
        in_channels: number of in channels for the block
        out_channels: number of out channels of the block
        kernel_size: kernel_size of the block
    """

    def __init__(self,in_channels:int,out_channels:int,kernel_size:int):
        super(ResNetBlock, self).__init__()
        self.projection_block = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.reslayer1 = ResNetLayer(kernel_size,out_channels,out_channels,dilation=1)
        self.reslayer2 = ResNetLayer(kernel_size,out_channels,out_channels,dilation=2)
        self.reslayer3 = ResNetLayer(kernel_size,out_channels,out_channels,dilation=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self,x):
        x = self.projection_block(x)
        x = self.maxpool(self.reslayer3(self.reslayer2(self.reslayer1(x))))
        return x



class FCLayer(nn.Module):

    """
    Fully connected layer for resnet module. Flattens the input and outputs a 128 size vector of embeddings.

    Arguments:
        in_channels: number of channels of the input
        shape_x,shape_y: shape of the input matrix

    """

    def __init__(self,in_channels:int,shape_x:int,shape_y:int):
        super(FCLayer, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=in_channels * shape_x * shape_y,out_features=128)

    def forward(self,x):
        x = self.fc1(self.flatten(x))
        return x
