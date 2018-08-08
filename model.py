import torch
import torch.nn as nn
from blocks import ConvBlock, ResnetBlock


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.f1 = ConvBlock(3, 96, kernel_size=3, padding=1)
        self.f2 = ConvBlock(96, 76, kernel_size=3, padding=1)
        self.f3 = ConvBlock(76, 65, kernel_size=3, padding=1)
        self.f4 = ConvBlock(65, 55, kernel_size=3, padding=1)

        self.rc_a = ConvBlock(292, 64, kernel_size=1, padding=0)
        self.rc_b1 = ConvBlock(292, 32, kernel_size=1, padding=0)
        self.rc_b2 = ConvBlock(32, 32, kernel_size=3, padding=3)

        self.rc_l = ConvBlock(96, 3, kernel_size=1, padding=0)

    def forward(self, input):

        f1x = self.f1(input)
        f2x = self.f2(f1x)
        f3x = self.f3(f2x)
        f4x = self.f4(f3x)

        x = torch.cat((f1x, f2x, f3x, f4x), 1)

        rca = self.rc_a(x)
        rcb = self.rc_b1(x)
        rcb = self.rc_b2(rcb)

        rc = torch.cat((rca, rcb), 1)

        lx = self.rc_l(rc)

        return lx


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.layers = nn.Sequential(
            ConvBlock(3, 64, kernel_size=3, padding=1),
            ConvBlock(3, 64, kernel_size=3, padding=1),
            
        )
        
    def forward(self, input):
        pass
