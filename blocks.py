import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride=stride, padding=padding)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'relu6':
            self.act = nn.ReLU6()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'elu':
            self.act = nn.ELU()
        elif self.activation == 'selu':
            self.act = nn.SELU()
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, activation='lrelu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'relu6':
            self.act = nn.ReLU6()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'elu':
            self.act = nn.ELU()
        elif self.activation == 'selu':
            self.act = nn.SELU()
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out


class BottleNeckBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, activation='relu', norm=None):
        super(BottleNeckBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter//4, 1, stride, 0)
        self.conv2 = torch.nn.Conv2d(num_filter//4, num_filter//4, kernel_size, stride, padding)
        self.conv3 = torch.nn.Conv2d(num_filter//4, num_filter, 1, stride, 0)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'relu6':
            self.act = nn.ReLU6()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'elu':
            self.act = nn.ELU()
        elif self.activation == 'selu':
            self.act = nn.SELU()
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv3(out))
        else:
            out = self.conv3(out)

        out = torch.add(out, residual)
        return out


class Pool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0, method='max'):
        super(Pool, self).__init__()

        if method == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif method == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return self.pool(x)
