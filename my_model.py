import torch
import torch.nn as nn
from blocks import ConvBlock, ResnetBlock


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        '''
        self.layers = nn.Sequential(
            ConvBlock(3, 64,  kernel_size=9, padding=4),
            ConvBlock(64, 32, kernel_size=1, padding=0),
            ConvBlock(32, 3,  kernel_size=5, padding=2, activation=None))
        '''
        self.layers = nn.Sequential(
            ConvBlock(3, 64, kernel_size=3, padding=1),
            ResnetBlock(64),
            ResnetBlock(64),
	    ResnetBlock(64),
            ResnetBlock(64),
	    ResnetBlock(64),
	    ResnetBlock(64),
            ConvBlock(64, 3, kernel_size=3, padding=1, activation=None))
    
    def forward(self, x):
        return (self.layers(x) + x)
