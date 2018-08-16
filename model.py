import torch
import torch.nn as nn
from torch.autograd import Variable

from blocks import ConvBlock, ResnetBlock, Pool, BottleNeckBlock, UpsampleBlock
from ms_ssim import msssim


class Generator(nn.Module):
    """
    Generate Super Resolution Image
    """
    def __init__(self, scale_factor=3):
        super(Generator, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor)

        layers = [ConvBlock(3, 128, kernel_size=9, padding=4, activation='prelu')]
        for i in range(5):
            layers.append(BottleNeckBlock(128, activation='prelu'))
        # layers.append(ConvBlock(128, scale_factor*scale_factor*15, kernel_size=1, padding=0, activation='prelu'))
        self.feature_ext = nn.Sequential(*layers)

        self.conv1 = ConvBlock(128, 3, kernel_size=3, padding=1, activation='prelu')
        self.upsample = UpsampleBlock(128, 3, upscale_factor=scale_factor)

        self.outconv = ConvBlock(6, 3, kernel_size=9, padding=4, activation=None)

    def forward(self, input):
        origin = self.conv1(self.feature_ext(input))
        half = self.upsample(self.feature_ext(self.pool(input)))
        return self.outconv(torch.cat((origin, half), 1))


class Discriminator(nn.Module):
    """
    Discriminator to discriminate SR and LR
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        layers = [
            ConvBlock(3, 32, kernel_size=3, padding=1, norm='batch', activation='lrelu'),
            ConvBlock(32, 64, kernel_size=3, padding=1, norm='batch', activation='lrelu'),
            ConvBlock(64, 64, kernel_size=3, stride=2, padding=1, activation='lrelu'),
            ConvBlock(64, 128, kernel_size=3, padding=1, norm='batch', activation='lrelu'),
            ConvBlock(128, 128, kernel_size=3, stride=2, padding=1, activation='lrelu'),
            ConvBlock(128, 256, kernel_size=3, padding=1, norm='batch', activation='lrelu'),
            ConvBlock(256, 256, kernel_size=3, stride=2, padding=1, activation='lrelu'),
            ConvBlock(256, 512, kernel_size=3, padding=1, norm='batch', activation='lrelu'),
            ConvBlock(512, 512, kernel_size=3, stride=2, padding=1, activation='lrelu'),

            ConvBlock(512, 1, kernel_size=1, stride=1, padding=1, activation=None),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.sigmoid(self.layers(x))
        return x


class SRLoss(nn.Module):
    """
    Multi losses
    """
    def __init__(self, loss='l1_mse', alpha=0.5):
        super(SRLoss, self).__init__()

        self.alpha = alpha

        if loss == 'l1_mse':
            self.l1 = nn.L1Loss()
            self.mse = nn.MSELoss()
            self.loss = self.l1_mse
        elif loss == 'msssim_l1':
            self.msssim = msssim
            self.l1 = nn.L1Loss()
            self.loss = self.msssim_l1

    def msssim_l1(self, input, target):
        return self.alpha * (1 - self.msssim(input, target)) + (1 - self.alpha) * (self.l1(input, target))

    def l1_mse(self, input, target):
        return self.alpha * self.l1(input, target) + (1-self.alpha) * self.mse(input, target)

    def forward(self, input, target):
        return self.loss(input, target)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, cuda=True):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = SRLoss(0.5)
        else:
            self.loss = nn.BCELoss()

        if cuda:
            self.loss = self.loss.cuda()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.cuda())
