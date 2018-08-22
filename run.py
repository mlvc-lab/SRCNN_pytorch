from __future__ import print_function
from os.path import join
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image

from torchvision.transforms import ToTensor
import numpy as np

from model import Generator

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--input_image', type=str, default='test.jpg', help='input image to use')
parser.add_argument('--output_filename', default='test_out.jpg', type=str, help='where to save the output image')
parser.add_argument('--scale_factor', default=3, type=float, help='factor by which super resolution needed')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--gpuids', default=[0], nargs='+', help='GPU ID for using')
opt = parser.parse_args()

opt.gpuids = list(map(int, opt.gpuids))
print(opt)

img = Image.open(opt.input_image).convert('YCbCr')
y, cb, cr = img.split()

img = img.resize((int(img.size[0]*opt.scale_factor), int(img.size[1]*opt.scale_factor)), Image.BICUBIC)

model_name = join("model", opt.model)
model = Generator()

with torch.no_grad():
    input = Variable(ToTensor()(img)).view(1, -1, img.size[1], img.size[0])

if opt.cuda:
    print("Using GPU")
    torch.cuda.set_device(opt.gpuids[0])
    input = input.cuda()
    model = nn.DataParallel(model, device_ids=opt.gpuids, output_device=opt.gpuids[0]).cuda()
    model.load_state_dict(torch.load(model_name))

else:
    print("Using CPU")
    state_dict = torch.load(model_name)
    state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        state_dict_rename[name] = v
    model.load_state_dict(state_dict_rename)


out = model(input)
out = torch.add(out, 1, input)  # residual learning
out = out.cpu()

print("type = ", type(out))
out_img_y = out.data[0].numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

# sharpen
# blurred = ndimage.gaussian_filter(out_img, 3)
# filter_blurred = ndimage.gaussian_filter(blurred, 1)
# alpha = 30
# sharpened = blurred + alpha * (blurred - filter_blurred)
# 
# imsave(opt.output_filename, sharpened)
out_img.save(opt.output_filename)
print('output image saved to ', opt.output_filename)
