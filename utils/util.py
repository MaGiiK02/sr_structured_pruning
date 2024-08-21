#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_tensor
from scipy.io import loadmat
from PIL import Image
import numpy as np
import math
from torchvision.utils import make_grid

#read an 8-bit image
def read_img(fname, grayscale=True):
    img = Image.open(fname)
    img = img.convert('L') if grayscale else img.convert('RGB')
    x = to_tensor(img)
    return x

#read an 8-bit/32-bit image in MATLAB format
def read_mat(fname,  grayscale=True, log_range=True):
    x = loadmat(fname, verify_compressed_data_integrity=False)['image']
    x = torch.FloatTensor(x)
    
    if (x.ndimension() == 3) and grayscale:
        x = x.transpose(2, 0)
        x = torch.sum(x, dim = 0) / 3
        x = x.unsqueeze(0)
    
    if log_range:  # perform log10(1 + image)
        x += 1
        torch.log10(x, out = x)
    elif x.ndimension() == 2:
        x = x.unsqueeze(0)
    return x

#read an image
def load_image(fname,  grayscale=True, log_range=True):
    filename, ext = os.path.splitext(fname)
    if ext == '.mat':
       return read_mat(fname, grayscale, log_range)
    else:
       return read_img(fname, grayscale)

#plot a graph with train, validation, and test
def plotGraph(array_train, array_val, array_test, folder):
    fig = plt.figure(figsize=(10, 4))
    n = min([len(array_train), len(array_val), len(array_test)])
    plt.plot(np.arange(1, n + 1), array_train[0:n])# train loss (on epoch end)
    plt.plot(np.arange(1, n + 1), array_val[0:n])  # val loss (on epoch end)
    plt.plot(np.arange(1, n + 1), array_test[0:n]) # test loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'validation','test'], loc="upper left")
    title = os.path.join(folder, "plot.png")
    plt.savefig(title, dpi=600)
    plt.close(fig)


def get_luminance(output):
    y_pred, y = output
    convert = y.new(1, 3, 1, 1)
    convert[0, 0, 0, 0] = 65.738
    convert[0, 1, 0, 0] = 129.057
    convert[0, 2, 0, 0] = 25.064
    y.mul_(convert)
    return y_pred.mul_(convert), y.mul_(convert)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

# From DRLN code
def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    '''
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    '''
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)