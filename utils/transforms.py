from numpy.core.fromnumeric import size
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import math
import random
import logging
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Sequence

# taken fromcvtorchvision
INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}
def rotateOpenCv(img, angle, resample='BILINEAR', expand=False, center=None):
    """Rotate the image by angle.
    Args:
        img (PIL Image): PIL Image to be rotated.
        angle ({float, int}): In degrees clockwise order.
        resample ({NEAREST, BILINEAR, BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """
    imgtype = img.dtype
    h, w, _ = img.shape
    point = center or (w/2, h/2)
    M = cv2.getRotationMatrix2D(point, angle=-angle, scale=1)

    if expand:
        if center is None:
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - point[0]
            M[1, 2] += (nH / 2) - point[1]

            # perform the actual rotation and return the image
            dst = cv2.warpAffine(img, M, (nW, nH))
        else:
            xx = []
            yy = []
            for point in (np.array([0, 0, 1]), np.array([w-1, 0, 1]), np.array([w-1, h-1, 1]), np.array([0, h-1, 1])):
                target = M@point
                xx.append(target[0])
                yy.append(target[1])
            nh = int(math.ceil(max(yy)) - math.floor(min(yy)))
            nw = int(math.ceil(max(xx)) - math.floor(min(xx)))
            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nw - w)/2
            M[1, 2] += (nh - h)/2
            dst = cv2.warpAffine(img, M, (nw, nh), flags=INTER_MODE[resample])
    else:
        dst = cv2.warpAffine(img, M, (w, h), flags=INTER_MODE[resample])
    return dst.astype(imgtype)

class DescreteRandomRotation:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)

        return TF.rotate(x, angle)

class DescreteRandomRotationCV2:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)

        return rotateOpenCv(x, angle)

def MosaicPatches(x, size_x:int, size_y: int):
    b, c, w, h = x.shape
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half, w_half 
    return [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

def RebuildMosaicPatches(x, size_x:int, size_y: int, unfolder):
    b, c, px, py,_, _ = unfolder
    unfold_shape = b, c, px, py, size_x, size_y
    patches_orig = x.view(unfold_shape)
    output_c = 3
    output_h = py * size_y
    output_w = px * size_x
    #reorder the tensor as b,c,w,px,h,py
    #patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3).contiguous()
    patches_orig = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
    # merge together w,px and h,py
    return patches_orig.view(1, output_c, output_h, output_w)

def paralellCrop(img1: np.ndarray, img2: np.ndarray, crop = 128, scale=1, random=None, cropInfo=None, cropMode=0): 
    if not random: random = np.random

    _, y, x = img1.shape
    
    cropX, cropY = None, None
    if cropInfo is None:
        if cropMode == 'random':
            cropX = int(random.random() * (x - crop))
            cropY = int(random.random() * (y - crop))
        else : #Center
            cropX = int(x/2 - crop/2)
            cropY = int(y/2 - crop/2)
    else:
        cropX, cropY = cropInfo

    cropXEnd = cropX + crop
    cropYEnd = cropY + crop

    cropXprarallel = cropX * scale
    cropYprarallel = cropY * scale
    cropXprarallelEnd = cropXprarallel + (crop * scale)
    cropYprarallelEnd = cropYprarallel + (crop * scale)
    
    img1crop = img1[:, cropY:cropYEnd, cropX:cropXEnd]
    img2crop = img2[:, cropYprarallel:cropYprarallelEnd, cropXprarallel:cropXprarallelEnd]
    
    return img1crop, img2crop, (cropX, cropY)