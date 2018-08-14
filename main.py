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
from model import Generator, Discriminator, GANLoss
from ms_ssim import ssim

# set option parameter
parser = argparse.ArgumentParser(description='PyTorch Super Resolution Example')
parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor")
parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=10, help='testing batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use')
parser.add_argument('--gpuids', default=[0], nargs='+', help='GPU ID for using')
parser.add_argument('--alpha', default=0.5, type=float, help='Loss alpha')
opt = parser.parse_args()

opt.gpuids = list(map(int, opt.gpuids))
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
generator = Generator()
discriminator = Discriminator()
criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

# set cuda(GPU)
if use_cuda:
    torch.cuda.set_device(opt.gpuids[0])
    with torch.cuda.device(opt.gpuids[0]):
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterionGAN = criterionGAN.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
    generator = nn.DataParallel(generator, device_ids=opt.gpuids, output_device=opt.gpuids[0])
    discriminator = nn.DataParallel(discriminator, device_ids=opt.gpuids, output_device=opt.gpuids[0])

g_optim = optim.Adam(generator.parameters(), lr=opt.lr)
d_optim = optim.Adam(discriminator.parameters(), lr=opt.lr)


# train
def train(epoch):
    """
    Training model
    take global epoch, then training one epoch

    :param epoch: global epoch
    :return: None
    """

    # epoch loss initialization
    epoch_d_loss = 0
    epoch_g_loss = 0

    # batch iteration
    for iteration, batch in enumerate(training_data_loader, 1):     # load training data from dataloader

        # set input and target data
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
            target = target - input   # residual

        ############# training D ##############
        if iteration % 2 == 0:
            d_optim.zero_grad()   # grad init
            gen_z = generator(input)    # gen SR image

            disc_z = discriminator(gen_z.detach())  # disc result of fake image
            fake_loss = criterionGAN(disc_z, False)  # gan fake loss

            disc_r = discriminator(target)  # disc result of real image
            real_loss = criterionGAN(disc_r, True)  # gan real loss

            loss_d = fake_loss + real_loss

            loss_d.backward()   # update grad
            d_optim.step()

            epoch_d_loss += loss_d

        ############# training G ##############
        g_optim.zero_grad()   # grad init
        gen_z = generator(input)    # sr
        disc_z = discriminator(gen_z)
        loss_g_gan = criterionGAN(disc_z, True)
        loss_g_l1 = criterionL1(gen_z, target)

        loss_g = (opt.alpha) * loss_g_gan + (1-opt.alpha) * loss_g_l1     # g loss
        loss_g.backward()   # update grad
        g_optim.step()

        epoch_g_loss = loss_g

    print("Epoch {}: GLoss: {:.6f} DLoss: {:.6f}"
          .format(epoch, epoch_g_loss / len(training_data_loader), epoch_d_loss / len(training_data_loader)))


def test():
    """
    Test model

    :return: None
    """
    avg_psnr = 0
    avg_ssim = 0
    for batch in testing_data_loader:
        with torch.no_grad():
            input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = generator(input)
        prediction = torch.add(prediction, 1, input)  # residual
        mse = nn.MSELoss()(prediction, target)
        psnr = 10 * log10(1 / mse.item())

        avg_psnr += psnr
        avg_ssim += ssim(target, prediction)

    avg_psnr /= len(testing_data_loader)
    avg_ssim /= len(testing_data_loader)

    print("===> Avg. PSNR: {:.4f} dB, SSIM: {:.4f}".format(avg_psnr, avg_ssim))


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

    model_out_path = "model/lr_{}_alpha_{}_epoch_{}.pth".format(opt.lr, opt.alpha, epoch)
    torch.save(generator.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


for epoch in range(1, opt.epochs + 1):
    train(epoch)
    test()
    if epoch % 10 == 0:
        checkpoint(epoch)

    print(opt)
