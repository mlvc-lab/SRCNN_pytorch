import torch
import torch.nn as nn
from blocks import ConvBlock, BottleneckBlock


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.layers = nn.Sequential(
            ConvBlock(3, 64,  kernel_size=3, padding=2),
            BottleneckBlock(64),
            BottleneckBlock(64),
            ConvBlock(64, 32, kernel_size=1),
            BottleneckBlock(32),
            BottleneckBlock(32),
            ConvBlock(32, 3,  kernel_size=3, activation=None))
    
    def forward(self, x):
        return self.layers(x)
