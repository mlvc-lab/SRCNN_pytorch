import torch
import torch.nn as nn
from blocks import ConvBlock, Conv_ReLU_Block





class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.input = ConvBlock(3,64,  kernel_size=3, padding=1)
        self.layers = self.make_layer(Conv_ReLU_Block, 8)
        self.output = ConvBlock(3,3, kernel_size=3, padding=1, activation=None)

            #ConvBlock(3, 256,  kernel_size=9, padding=4),
            #ResnetBlock(256),
            #ConvBlock(256, 3,  kernel_size=5, padding=2, activation=None)
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.input.forward(x)
        out = self.layers(out)
        out = self.output.forward(out)
        out2 = torch.add(out,residual)
        return out, out2
