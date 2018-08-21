import torch
import torch.nn as nn
from blocks import ConvBlock


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.layers = nn.Sequential(
            ConvBlock(3, 256,  kernel_size=1, padding=0),
            ConvBlock(256, 128, kernel_size=1, padding=0),
            ConvBlock(128, 3,  kernel_size=1, padding=0, activation=None))
    
    def forward(self, x):
        return self.layers(x)
