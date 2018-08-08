import torch
import torch.nn as nn
from blocks import ConvBlock, ResnetBlock


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.layers = nn.Sequential(
            ConvBlock(3, 64,  kernel_size=3, padding=1),
	    ConvBlock(64, 128, kernel_size=3, padding=1),
	    ResnetBlock(128),
	    ResnetBlock(128),
	    ConvBlock(128, 64, kernel_size=3, padding=1),
            ConvBlock(64, 32, kernel_size=3, padding=1),
            ConvBlock(32, 3,  kernel_size=1, padding=0, activation=None))
    
    def forward(self, x):
        return self.layers(x)
