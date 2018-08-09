import torch
import torch.nn as nn

from blocks import ConvBlock, ResnetBlock
from ms_ssim import msssim


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.f1 = ConvBlock(3, 96, kernel_size=3, padding=1)
        self.f2 = ConvBlock(96, 76, kernel_size=3, padding=1)
        self.f3 = ConvBlock(76, 65, kernel_size=3, padding=1)
        self.f4 = ConvBlock(65, 55, kernel_size=3, padding=1)
        self.f5 = ConvBlock(55, 47, kernel_size=3, padding=1)
        self.f6 = ConvBlock(47, 39, kernel_size=3, padding=1)
        self.f7 = ConvBlock(39, 32, kernel_size=3, padding=1)

        self.rc_a = ConvBlock(410, 64, kernel_size=1, padding=0)
        self.rc_b1 = ConvBlock(410, 32, kernel_size=1, padding=0)
        self.rc_b2 = ConvBlock(32, 32, kernel_size=3, padding=1)

        self.rc_l = ConvBlock(96, 16, kernel_size=1, padding=0)
        self.sp = ConvBlock(16, 3, kernel_size=5, padding=2)

    def forward(self, input):

        f1x = self.f1(input)
        f2x = self.f2(f1x)
        f3x = self.f3(f2x)
        f4x = self.f4(f3x)
        f5x = self.f5(f4x)
        f6x = self.f6(f5x)
        f7x = self.f7(f6x)

        x = torch.cat((f1x, f2x, f3x, f4x, f5x, f6x, f7x), 1)

        rca = self.rc_a(x)
        rcb = self.rc_b1(x)
        rcb = self.rc_b2(rcb)

        rc = torch.cat((rca, rcb), 1)

        lx = self.rc_l(rc)
        spx = self.sp(lx)

        return spx


class SRLoss(nn.Module):
    def __init__(self, alpha):
        super(SRLoss, self).__init__()

        self.alpha = alpha
        self.msssim = msssim
        self.l1 = nn.L1Loss()

    def forward(self, input, target):
        return self.alpha * (1 - self.msssim(input, target)) + (1 - self.alpha) * (self.l1(input, target))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            ConvBlock(3, 64, kernel_size=3, padding=1),
            ConvBlock(3, 64, kernel_size=3, padding=1),
        )

    def forward(self, input):
        pass
