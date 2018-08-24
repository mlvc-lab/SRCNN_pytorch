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
from model import SRCNN, Generator, init_weights, _NetG, SRLoss
from torchvision import models
import torch.utils.model_zoo as model_zoo


parser = argparse.ArgumentParser(description='PyTorch Super Resolution Example')
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=10, help='testing batch size')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--gpuids', default=[1,2,3], nargs='+',  help='GPU ID for using')
parser.add_argument('--alpha', default=0.5, type=float, help='Loss alpha')
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
# parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

opt.gpuids = list(map(int,opt.gpuids))
print(opt)


use_cuda = opt.cuda
if use_cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# torch.manual_seed(opt.seed)
# if use_cuda:
#     torch.cuda.manual_seed(opt.seed)


train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)


if opt.vgg_loss:
        print('===> Loading VGG model')
        netVGG = models.vgg19()
        netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))
        class _content_model(nn.Module):
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])

            def forward(self, x):
                out = self.feature(x)
                return out

        netContent = _content_model()



model = Generator()
#srcnn = _NetG()
criterion = nn.MSELoss()
#criterion = SRLoss(opt.alpha)

if(use_cuda):
        torch.cuda.set_device(opt.gpuids[0])
        model = model.cuda()
        criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=opt.gpuids, output_device=opt.gpuids[0]).cuda()
        if opt.vgg_loss:
            netContent = netContent.cuda()


#optimizer = optim.SGD(srcnn.parameters(),lr=opt.lr, momentum=0.9,weight_decay=0.001)
optimizer = optim.Adam(model.parameters(),lr=opt.lr)
#optimizer = optim.SGD(srcnn.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
        

        optimizer.zero_grad()
        model_out = model(input)

        if opt.vgg_loss:
            content_input = netContent(model_out)
            content_target = netContent(target)
            content_target = content_target.detach()
            content_loss = criterion(content_input, content_target)

        if opt.vgg_loss:
            netContent.zero_grad()
            content_loss.backward(retain_graph=True)

        
        loss = criterion(model_out, target)



        epoch_loss = epoch_loss + loss.item()
        loss.backward()
       # nn.utils.clip_grad_norm(model.parameters(),0.25)
        optimizer.step()

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))




def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = model(input)

        mse = nn.MSELoss()(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    return (avg_psnr / len(testing_data_loader))

def checkpoint(epoch, psnr):
    try:
        if not(os.path.isdir('model')):
            os.makedirs(os.path.join('model'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    model_out_path = "model/model_epoch_{}_psnr_{:.4f}.pth".format(epoch,psnr )
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, opt.epochs + 1):
    train(epoch)
    psnr = test()
    checkpoint(epoch, psnr)
