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
        if self.activation is not None:
            return self.act(self.conv(x))
        else:
            return self.conv(x)



class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, activation='relu', norm='batch'):
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
           # out = self.conv2(out)
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
            super(Conv_ReLU_Block, self).__init__()
            self.bn = torch.nn.BatchNorm2d(64)
            self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu = nn.ReLU(True)

    def forward(self, x):
            return self.relu(self.conv(x))

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class residualBlock(nn.Module):
    def __init__(self, in_channels=128, k=3, n=32, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n , kernel_size=1, stride=s)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)
        self.conv3 = nn.Conv2d(n, in_channels, kernel_size=1, stride=s)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.PReLu = nn.PReLU(init=0.2)

    def forward(self, x):
        y =self.PReLu(self.bn1(self.conv1(x)))
        y =self.PReLu(self.bn2(self.conv2(y)))
        y =self.bn3(self.conv3(y))
        y += x
        y = self.PReLu(y)
        return y




class ResNetPreActBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, activation='relu', norm=None):
        super(ResNetPreActBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter//4, 1, stride, 0)
        self.conv2 = torch.nn.Conv2d(num_filter//4, num_filter//4, kernel_size, stride, padding)
        self.conv3 = torch.nn.Conv2d(num_filter//4, num_filter, 1, stride, 0)

        self.norm = norm
        if self.norm == 'batch':
            self.bn1 = torch.nn.BatchNorm2d(num_filter)
            self.bn2 = torch.nn.BatchNorm2d(num_filter//4)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(inplace=False)
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

    def forward(self, out):
        residual = out

        # 1
        if self.norm is not None:
            out = self.bn1(out)

        if self.activation is not None:
            out = self.act(out)

        out = self.conv1(out)

        # 2
        if self.norm is not None:
            out = self.bn2(out)

        if self.activation is not None:
            out = self.act(out)

        out = self.conv2(out)

        # 3
        if self.norm is not None:
            out = self.bn2(out)

        if self.activation is not None:
            out = self.act(out)

        out = self.conv3(out)

        # residual
        out = torch.add(out, residual)
        return out


