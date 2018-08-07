import torch
import torch.nn as nn
from blocks import ConvBlock


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.input = ConvBlock(3, 64,  kernel_size=9, padding=1, norm='batch')
        self.layers = self.make_layer(ConvBlock(64, 64, kernel_size=3, padding=1, norm='batch'), 17)
        self.secondout = ConvBlock(64, 32, kernel_size=3, padding=1, norm='batch')
        self.out = ConvBlock(32, 3,  kernel_size=3, padding=1, norm='batch', activation=None)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        x= self.input.forward(x)
        x = self.layers(x)
        x = self.secondout.forward(x)
        x = self.out.forward(x)
        x = torch.add(residual, x)
        return x

