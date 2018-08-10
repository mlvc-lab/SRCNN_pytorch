import torch
import torch.nn as nn

from blocks import ConvBlock, ResnetBlock
from ms_ssim import msssim


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        layers = [ConvBlock(3, 32, kernel_size=5, padding=2), ConvBlock(32, 64, kernel_size=5, padding=2)]
        for i in range(8):
            layers.append(ResnetBlock(64))
        layers.append(ConvBlock(64, 3, kernel_size=3, padding=1, activation=None))
        layers.append(ConvBlock(3, 3, kernel_size=1, padding=0, activation=None))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class SRLoss(nn.Module):
    def __init__(self, alpha):
        super(SRLoss, self).__init__()

        self.alpha = alpha
        self.msssim = msssim
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def msssim_l1(self, input, target):
        return self.alpha * (1 - self.msssim(input, target)) + (1 - self.alpha) * (self.l1(input, target))

    def l1_mse(self, input, target):
        return self.alpha * self.l1(input, target) + (1-self.alpha) * self.mse(input, target)

    def forward(self, input, target):
        return self.l1_mse(input, target)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            ConvBlock(3, 64, kernel_size=3, padding=1),
            ConvBlock(3, 64, kernel_size=3, padding=1),
        )

    def forward(self, input):
        pass
