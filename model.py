import torch
import torch.nn as nn
from blocks import ConvBlock, Conv_ReLU_Block, ResnetBlock, Bottleneck, ResNetPreActBlock, residualBlock
from math import sqrt
import math

from ms_ssim import msssim




class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

       # self.input = ConvBlock(3,64,  kernel_size=3, padding=1)
       # self.layers = self.make_layer(Conv_ReLU_Block, 4)
       # self.output = ConvBlock(64,3, kernel_size=3, padding=1, activation=None)
       # for m in self.modules():
       #     if isinstance(m, nn.Conv2d):
       #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
       #         m.weight.data.normal_(0, sqrt(2. / n))
       # self.layers = nn.Sequential(
       # ConvBlock(3,64, kernel_size=9, padding=4),
       # ResnetBlock(64, kernel_size=3, padding=1),
       # Bottleneck(64,64),
       # ConvBlock(256,3,kernel_size=3, padding=1,activation=None)
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = torch.nn.BatchNorm2d(64)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn(self.input(x)))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.BatchNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output
 
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class upsampleBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=576):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(3)
        self.prelu = nn.PReLU(64)
    def forward(self, x):
        return self.prelu(self.shuffler(self.conv(x)))



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.input = ConvBlock(3, 128, kernel_size=9, padding=4)
        self.Prelu = nn.PReLU(init=0.2)

        self.layers = []
        for i in range(8):
            self.layers.append(residualBlock())
        self.reallayers = nn.Sequential(*self.layers)
        
        self.inlayer = ConvBlock(128,128,kernel_size=3,padding=1)
        
        self.layers2=[]
        for i in range(2):
            self.layers.append(upsampleBlock())
        self.reallayers2 = nn.Sequential(*self.layers2)

        self.output = ConvBlock(128, 32, kernel_size=3, padding=1, activation=None, norm=None)
        self.lastoutput = ConvBlock(32,3,kernel_size=3, padding=1, activation=None, norm=None)
    def forward(self, x):
        residual = self.Prelu(self.input(x))
        out = self.reallayers(residual)
        out = self.inlayer(out) + residual
        out = self.reallayers2(out)
        out = self.output(out)
        out = self.lastoutput(out)
        return out

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

    def l2tol1(self, input, target, global_epoch, current_epoch):
        if current_epoch > global_epoch // 2:
            return self.l1(input, target)
        return self.mse(input, target)

    def forward(self, input, target, global_epoch, current_epoch):
        return self.l2tol1(input, target, global_epoch, current_epoch)



class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        #self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 




class _NetG(nn.Module):
    def __init__(self):
        super(_NetG, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        out = self.conv_output(out)
        return out
