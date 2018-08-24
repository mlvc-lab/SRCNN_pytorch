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

from skimage.measure import compare_ssim as ssim

from data import get_training_set, get_test_set
from model import Generator, Discriminator, GANLoss, SRLoss
# from ms_ssim import SSIM

# set option parameter
parser = argparse.ArgumentParser(description='PyTorch Super Resolution Example')
parser.add_argument('--save_path', type=str, default='model', help='model save path')
parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor")
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
parser.add_argument('--gpuids', default=[0, 1, 2, 3], nargs='+', help='GPU ID for using')
parser.add_argument('--alpha', default=0.75, type=float, help='Loss alpha')
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
criterionGAN = GANLoss(alpha=opt.alpha)
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()
# criterionSSIM = SSIM()

# set cuda(GPU)
if use_cuda:
    torch.cuda.set_device(opt.gpuids[0])
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    # criterionSSIM = criterionSSIM.cuda()
    # set DataParallel to use multi gpu
    generator = nn.DataParallel(generator, device_ids=opt.gpuids, output_device=opt.gpuids[0]).cuda()
    discriminator = nn.DataParallel(discriminator, device_ids=opt.gpuids, output_device=opt.gpuids[0]).cuda()

g_optim = optim.Adam(generator.parameters(), lr=opt.lr)
d_optim = optim.Adam(discriminator.parameters(), lr=opt.lr)
# g_optim = optim.RMSprop(generator.parameters(), lr=opt.lr)
# d_optim = optim.RMSprop(discriminator.parameters(), lr=opt.lr)


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
        if use_cuda:    # gpu
            input = input.cuda()
            target = target.cuda()
        target = target - input   # residual

        #######################################
        # training D
        #######################################
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

        #######################################
        # training G
        #######################################
        g_optim.zero_grad()   # grad init
        gen_z = generator(input)    # sr
        disc_z = discriminator(gen_z)
        id_target = generator(target)

        # calculate loss
        loss_g_gan = criterionGAN(disc_z, True)
        loss_g_l1 = criterionL1(gen_z, target)
        # loss_ssim = criterionSSIM(gen_z, target)
        loss_id_l1 = criterionL1(id_target, target)

        loss_g = loss_g_gan + loss_g_l1 + loss_id_l1    # g loss
        loss_g.backward()   # update grad
        g_optim.step()

        epoch_g_loss = loss_g

    # print result
    print("Epoch {}: GLoss: {:.6f} DLoss: {:.6f}"
          .format(epoch, epoch_g_loss / len(training_data_loader), epoch_d_loss / len(training_data_loader)))


def test():
    """
    Test model
    print psnr of current model

    TODO if you want to use ssim plsea uncomment 3 lines but it can rise OOM

    :return: avg PSNR
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

        # PSNR
        mse = nn.MSELoss()(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr

        # SSIM
        for i in range(opt.test_batch_size):
            avg_ssim += ssim(target[i].permute(1, 2, 0).cpu().detach().numpy(),
                             prediction[i].permute(1, 2, 0).cpu().detach().numpy(),
                             multichannel=True)
        # avg_ssim += criterionSSIM(target, prediction)

    avg_psnr /= len(testing_data_loader)
    avg_ssim /= len(testing_data_loader)*opt.test_batch_size

    print("===> Avg. PSNR: {:.4f} dB, SSIM: {:.4f}".format(avg_psnr, avg_ssim))

    return avg_psnr


def checkpoint(epoch, psnr):
    """
    Save checkpoint name of '[save_path]/model_epoch_<epoch>.pth'

    :param epoch: epoch of current trained model
    :return: None
    """
    try:
        if not (os.path.isdir(opt.save_path)):
            os.makedirs(os.path.join(opt.save_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    model_out_path = "{}/psnr_{:.4f}_lr_{}_alpha_{:.2f}_epoch_{}.pth" \
                     .format(str(opt.save_path), psnr, opt.lr, opt.alpha, epoch)
    torch.save(generator.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def main():
    """
    Training and validating model
    :return: last psnr of model
    """
    for epoch in range(1, opt.epochs + 1):
        train(epoch)
        psnr = 0
        if epoch % 10 == 0:
            psnr = test()
        if epoch % 10 == 0:
            checkpoint(epoch, psnr)

    return psnr


def repeate_check(lrs=(0.0003, 0.0001), alphas=(0.2, 0.3), result_file='result.txt'):
    """
    Repeate check for experiment

    :param lrs: learning-rate list or tuple
    :param alphas: alpha list or tuple
    :param result_file: file path to save result
    :return: None
    """

    avg_psnr = 0

    f = open("result.txt", "w+")

    for lr in lrs:
        opt.lr = lr
        for alpha in alphas:
            opt.alpha = alpha
            opt.save_path = 'model_{}'.format(iter)
            print(opt)
            avg_psnr += main()

            f.write("lr:{:.5}\talpha:{:.2}\tPSNR:{:.4}".format(lr, alpha, avg_psnr/5))
    f.close()


if __name__ == '__main__':

    main()
