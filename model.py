import torch
import torch.nn as nn
from blocks import ConvBlock,ResnetBlock

import collections


class SRCNN(nn.Module):
    def __init__(self, kernel_size = [9, 1, -1, 5], channel = 64, batchuse = False, Res = False):
        super(SRCNN, self).__init__()
        if Res == False:
            if batchuse == False:

                sequentailList = collections.OrderedDict()
                sequentailList['conv1'] = ConvBlock(3, channel,  kernel_size=kernel_size[0], padding=(kernel_size[0] - 1) / 2)

                if kernel_size[1] != -1:
                    sequentailList['conv2'] = ConvBlock(channel, channel/2, kernel_size=kernel_size[1], padding=(kernel_size[1] - 1) / 2)
                    channel = channel / 2
                if kernel_size[2] != -1:
                    sequentailList['conv3'] = ConvBlock(channel, channel / 2, kernel_size=kernel_size[2], padding=(kernel_size[2] - 1) / 2)
                    channel = channel / 2

                sequentailList['conv4'] = ConvBlock(channel, 3, kernel_size=kernel_size[3], padding=(kernel_size[3] - 1) / 2, activation=None)
                self.layers = nn.Sequential(sequentailList)

            if batchuse == True:

                sequentailList = collections.OrderedDict()
                sequentailList['conv1'] = ConvBlock(3, channel, kernel_size=kernel_size[0],
                                                    padding=(kernel_size[0] - 1) / 2 , norm='batch')

                if kernel_size[1] != -1:
                    sequentailList['conv2'] = ConvBlock(channel, channel / 2, kernel_size=kernel_size[1],
                                                        padding=(kernel_size[1] - 1) / 2 , norm='batch')
                    channel = channel / 2
                if kernel_size[2] != -1:
                    sequentailList['conv3'] = ConvBlock(channel, channel / 2, kernel_size=kernel_size[2],
                                                        padding=(kernel_size[2] - 1) / 2 , norm='batch')
                    channel = channel / 2

                sequentailList['conv4'] = ConvBlock(channel, 3, kernel_size=kernel_size[3],
                                                    padding=(kernel_size[3] - 1) / 2, activation=None , norm='batch')
                self.layers = nn.Sequential(sequentailList)

        if Res == True:
            sequentailList = collections.OrderedDict()
            sequentailList['res1'] = ResnetBlock(3, channel, kernel_size=kernel_size[0],
                                                padding=(kernel_size[0] - 1) / 2)

            if kernel_size[1] != -1:
                sequentailList['res2'] = ResnetBlock(channel, channel / 2, kernel_size=kernel_size[1],
                                                    padding=(kernel_size[1] - 1) / 2)
                channel = channel / 2
            if kernel_size[3] != -1:
                sequentailList['conv3'] = ConvBlock(channel, 3, kernel_size=kernel_size[2],
                                                    padding=(kernel_size[2] - 1) / 2, activation=None)

            self.layers = nn.Sequential(sequentailList)


    def forward(self, x):
        return self.layers(x)
