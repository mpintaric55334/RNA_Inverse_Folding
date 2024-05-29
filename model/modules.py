import torch.nn as nn


class ResNetLayer(nn.Module):

    """
    Basic building block of the resnet model.
    Consists of two convolution layers,
    each layer having same dilation,and same kernel size.
    After each layer, batchnorm and relu are done.

    Arguments:
        kernel_size: size of the kernel
        channels: number of channels
        dilation: dilation number
    """

    def __init__(self, kernel_size: int, channels: int, dilation: int):

        super(ResNetLayer, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size,
                               padding="same", dilation=dilation)
        self.batchnorm1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size,
                               padding="same", dilation=dilation)
        self.batchnorm2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):

        x_input = x
        x = self.relu1(self.batchnorm1(self.conv1(x)))
        x = self.relu2(self.batchnorm2(self.conv2(x)))
        x = x + x_input
        return x


class ResNetBlock(nn.Module):

    """
    Larger block of the module, consists of three resnet layers,
    so all 3 dilations are used in one block.
    Each resnet layer
    has same number of in and out channels as 
    the others in the block. At the end of the block, max pooling is done.

    Arguments:
        channels: number of channels for the block
        kernel_size: kernel_size of the block
    """

    def __init__(self, channels: int, kernel_size: int):
        super(ResNetBlock, self).__init__()
        self.reslayer1 = ResNetLayer(kernel_size, channels, dilation=1)
        self.reslayer2 = ResNetLayer(kernel_size, channels, dilation=2)
        self.reslayer3 = ResNetLayer(kernel_size, channels, dilation=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.maxpool(self.reslayer3(self.reslayer2(self.reslayer1(x))))
        return x


class FCLayer(nn.Module):

    """
    Fully connected layer for resnet module. Flattens the input and
    outputs a 128 size vector of embeddings.

    Arguments:
        in_channels: number of channels of the input
        shape_x,shape_y: shape of the input matrix

    """

    def __init__(self, in_channels: int, shape_x: int, shape_y: int):
        super(FCLayer, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=in_channels * shape_x * shape_y,
                             out_features=512)

    def forward(self, x):
        x = self.fc1(self.flatten(x))
        return x


class Encoder(nn.Module):

    """
    Encoder(resnet) class.

    Arguments:
        matrix_shape: input matrix shape

    """

    def __init__(self, in_channels: int = 1, desired_channels: int = 128,
                 matrix_shape: int = 512):
        
        super(Encoder, self).__init__()
        self.projection_block = nn.Conv2d(in_channels, desired_channels,
                                          kernel_size=1)
        self.blocks = nn.ModuleList()
        for _ in range(5):
            self.blocks.append(ResNetBlock(channels=desired_channels,
                                           kernel_size=3))
            matrix_shape = int(matrix_shape/2)

        self.blocks.append(FCLayer(in_channels=desired_channels,
                                   shape_x=matrix_shape, shape_y=matrix_shape))

    def forward(self, x):
        x = self.projection_block(x)
        for block in self.blocks:
            x = block(x)
        return x


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=20, hidden_size=hidden_size,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 6)  # hidden_size, class number
        self.embedding = nn.Embedding(7, 20, padding_idx=6)

    def forward(self, inputs, states):
        embeddings = self.embedding(inputs.long())
        hiddens, new_states = self.lstm(embeddings, states)
        outputs = self.fc(hiddens)
        return outputs, new_states
