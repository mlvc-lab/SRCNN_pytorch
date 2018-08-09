from __future__ import print_function
import argparse
from math import log10
import os
from os import errno

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
from model import SRCNN, SRLoss

# set option parameter
parser = argparse.ArgumentParser(description='PyTorch Super Resolution Example')
parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor")
parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=10, help='testing batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use')
parser.add_argument('--gpuid', default=0, type=int, help='GPU ID for using')
opt = parser.parse_args()

print(opt)

# cuda(GPU) exception
use_cuda = opt.cuda
if use_cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# load dataset, dataloader
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size,
                                 shuffle=False)

# load model and criterion(loss)
srcnn = SRCNN()
criterion = SRLoss(0.84)

# set cuda(GPU)
if (use_cuda):
    torch.cuda.set_device(opt.gpuid)
    srcnn.cuda()
    criterion = criterion.cuda()

# optimizer = optim.SGD(srcnn.parameters(),lr=opt.lr)
optimizer = optim.Adam(srcnn.parameters(), lr=opt.lr)


# train
def train(epoch):
    """
    Training model
    take global epoch, then training one epoch

    :param epoch: global epoch
    :return: None
    """

    # epoch loss initialization
    epoch_loss = 0

    # batch iteration
    for iteration, batch in enumerate(training_data_loader, 1):     # load training data from dataloader
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        model_out = srcnn(input)
        loss = criterion(model_out, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        # if (iteration % 10) == 1:
        #     print(
        #         "===> Epoch[{}]({}/{}): Loss: {:.6f}"
        #          .format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    """
    Test model

    :return: None
    """
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = srcnn(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.6f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    """
    Save checkpoint name of 'model_epoch_<epoch>.pth' 

    :param epoch: epoch of current trained model
    :return: None
    """
    try:
        if not (os.path.isdir('model')):
            os.makedirs(os.path.join('model'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    model_out_path = "model/model_epoch_{}.pth".format(epoch)
    torch.save(srcnn, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


for epoch in range(1, opt.epochs + 1):
    train(epoch)
    test()
    if epoch % 10 == 0:
        checkpoint(epoch)
