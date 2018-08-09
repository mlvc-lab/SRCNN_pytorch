import torch
import torch.nn as nn
from blocks import ConvBlock, ResnetBlock, Conv_ReLU_Block





class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.input = ConvBlock(3, 64,  kernel_size=3, padding=1, norm='batch')
        self.layers = self.make_layer(Conv_ReLU_Block, 8)
        self.output = ConvBlock(3,3, kernel_size=3, padding=1, activation=None, norm='batch')

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
        out = self.output.forward(x)
        out = torch.add(out,residual)
        return out

class DRCN(nn.Module):
    def __init__(self, n_class):
        super(DRCN, self).__init__()

        # convolutional encoder

        self.enc_feat = nn.Sequential()
        self.enc_feat.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=100, kernel_size=5,
                                                    padding=2))
        self.enc_feat.add_module('relu1', nn.ReLU(True))
        self.enc_feat.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.enc_feat.add_module('conv2', nn.Conv2d(in_channels=100, out_channels=150, kernel_size=5,
                                                    padding=2))
        self.enc_feat.add_module('relu2', nn.ReLU(True))
        self.enc_feat.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.enc_feat.add_module('conv3', nn.Conv2d(in_channels=150, out_channels=200, kernel_size=3,
                                                    padding=1))
        self.enc_feat.add_module('relu3', nn.ReLU(True))

        self.enc_dense = nn.Sequential()
        self.enc_dense.add_module('fc4', nn.Linear(in_features=200 * 8 * 8, out_features=1024))
        self.enc_dense.add_module('relu4', nn.ReLU(True))
        self.enc_dense.add_module('drop4', nn.Dropout2d())

        self.enc_dense.add_module('fc5', nn.Linear(in_features=1024, out_features=1024))
        self.enc_dense.add_module('relu5', nn.ReLU(True))

        # label predict layer
        self.pred = nn.Sequential()
        self.pred.add_module('dropout6', nn.Dropout2d())
        self.pred.add_module('predict6', nn.Linear(in_features=1024, out_features=n_class))

        # convolutional decoder

        self.rec_dense = nn.Sequential()
        self.rec_dense.add_module('fc5_', nn.Linear(in_features=1024, out_features=1024))
        self.rec_dense.add_module('relu5_', nn.ReLU(True))

        self.rec_dense.add_module('fc4_', nn.Linear(in_features=1024, out_features=200 * 8 * 8))
        self.rec_dense.add_module('relu4_', nn.ReLU(True))

        self.rec_feat = nn.Sequential()

        self.rec_feat.add_module('conv3_', nn.Conv2d(in_channels=200, out_channels=150,
                                                     kernel_size=3, padding=1))
        self.rec_feat.add_module('relu3_', nn.ReLU(True))
        self.rec_feat.add_module('pool3_', nn.Upsample(scale_factor=2))

        self.rec_feat.add_module('conv2_', nn.Conv2d(in_channels=150, out_channels=100,
                                                     kernel_size=5, padding=2))
        self.rec_feat.add_module('relu2_', nn.ReLU(True))
        self.rec_feat.add_module('pool2_', nn.Upsample(scale_factor=2))

        self.rec_feat.add_module('conv1_', nn.Conv2d(in_channels=100, out_channels=3,
                                                     kernel_size=5, padding=2))

    def forward(self, input_data):
        feat = self.enc_feat(input_data)
        feat = feat.view(-1, 200 * 8 * 8)
        feat_code = self.enc_dense(feat)

        pred_label = self.pred(feat_code)

        feat_encode = self.rec_dense(feat_code)
        feat_encode = feat_encode.view(-1, 200, 8, 8)
        img_rec = self.rec_feat(feat_encode)

        return pred_label, img_rec


