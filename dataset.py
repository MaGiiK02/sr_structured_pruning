# Prepare the dataset for the training phase

import torch.utils.data as data
import torchvision.transforms as T
import glob2
import torch
import random
import numpy as np
import os.path as path
from utils.transforms import paralellCrop
from PIL import Image
import imageio
import skimage.color as sc


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

def frame_to_tensor(input_image):
    img_array = np.asarray(input_image).copy().astype(np.float)
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute((2, 1, 0))
    return img_tensor

# To handle grayscale images by inllating the gray channel
def set_channel(img, n_channel):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    c = img.shape[2]
    if n_channel == 1 and c == 3:
        img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
    elif n_channel == 3 and c == 1:
        img = np.concatenate([img] * n_channel, 2)

    return img


def np2Tensor(img, rgb_range):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor


class Banchmark(data.Dataset):
    data_path = '/trinity/home/mangelini/data/mangelini/Pruning/Dataset/eval/benchmark'
    
    def __init__(self, scaling, useBGR=False, take=None, shuffle=False):
        super(Banchmark, self).__init__()
        if not (scaling in [2, 3, 4, 8]): raise Exception(f'Banchmark have no dataset for scaling factor {scaling} -> Accepted:[2, 3, 4, 8]')
        self.scaling = scaling
        self.useBGR = useBGR
        self.original_filenames = glob2.glob(f'{self.data_path}/**/HR/*.png')
        random.seed(0)
        if shuffle: random.shuffle(self.original_filenames)
        if take: self.original_filenames = self.original_filenames[:take]

    def _getMax(self):
        return 255

    def getDataRange(self):
        return 255

    def load(self, path_img):
        # img = Image.open(path_img).convert("RGB")
        img = imageio.imread(path_img)
        img = set_channel(img, 3)
        if self.useBGR: img = img[:, :, ::-1]
        return img

    def prepare(self, data):
        # data = frame_to_tensor(data)
        data = np2Tensor(data, 255)
        return data

    def getSequenceName(self, img_path):
        return img_path.split('/')[-4]
    
    def getSourcePath(self, img_path):
        return img_path.replace('/HR/', f'/LR_bicubic/X{self.scaling}/').replace('.png',f'x{self.scaling}.png')

    def __getitem__(self, index):
        target_path = self.original_filenames[index]
        target = self.load(target_path)
        target_h = target.shape[0]
        target_w = target.shape[1]
        
        input_path = self.getSourcePath(target_path)
        input = self.load(input_path)
        input_h = input.shape[0]
        input_w = input.shape[1]
    
        #Fix for images with strange sizes
        if target_w%self.scaling !=0 :
            target = target[0:target_h, 0:target_w-(target_w%self.scaling)]
        if target_h%self.scaling !=0 :
            target = target[0:target_h-(target_h%self.scaling), 0:target_w]
        target_h = target.shape[0]
        target_w = target.shape[1]
            
        if input_w*self.scaling != target_w and input_h*self.scaling != target_h:
            raise Exception(f"Super Resolution pairs sizes dose not match ({input_w},{input_h})->({input_w*self.scaling},{input_h*self.scaling}) but expected ({target_w/self.scaling},{target_h/self.scaling})->({target_w},{target_h}).\n Paths: {input_path}->{target_path}")
        
        input = self.prepare(input)
        target = self.prepare(target) 
        
        return (input, target, {
            "sequence_name": self.getSequenceName(input_path),
            "frame": path.basename(input_path).replace(".png", "")
        })

    def __len__(self):
        return len(self.original_filenames)
    

class Div2KTrain(data.Dataset):
    data_path = '/trinity/home/mangelini/data/mangelini/Pruning/Dataset/train/DIV2K2017/DIV2K'
    
    def __init__(self, scaling, useBGR=False):
        super(Div2KTrain, self).__init__()
        if not (scaling in [2, 3, 4, 8]): raise Exception(f'DIV2K have no dataset for scaling factor {scaling} -> Accepted:[2, 3, 4, 8]')
        self.scaling = scaling
        self.useBGR = useBGR
        self.original_filenames = glob2.glob(f'{self.data_path}/DIV2K_train_HR/*.png')


    def _getMax(self):
        return 255

    def getDataRange(self):
        return 255

    def load(self, path_img):
        img = imageio.imread(path_img)       
        img = set_channel(img, 3)
        if self.useBGR: img = img[:, :, ::-1]
        return img

    def prepare(self, data):
        # data = frame_to_tensor(data)
        data = np2Tensor(data, 255)
        return data

    def getSequenceName(self, img_path):
        return img_path.split('/')[-4]
    
    def getSourcePath(self, img_path):
        return img_path.replace('/DIV2K_train_HR/', f'/DIV2K_train_LR_bicubic/X{self.scaling}/').replace('.png',f'x{self.scaling}.png')

    def __getitem__(self, index):
        target_path = self.original_filenames[index]
        target = self.load(target_path)
        target_h = target.shape[0]
        target_w = target.shape[1]
        
        input_path = self.getSourcePath(target_path)
        input = self.load(input_path)
        input_h = input.shape[0]
        input_w = input.shape[1]
    
        #Fix for images with Odd sizes
        if target_w%2 ==1 :
            target = target[0:target_h, 0:target_w-1]
        if target_h%2 ==1 :
            target = target[0:target_h-1, 0:target_w]
        target_h = target.shape[0]
        target_w = target.shape[1]
        
        # if input_w*self.scaling != target_w and input_h*self.scaling != target_h:
        #     raise Exception(f"Super Resolution pairs sizes dose not match ({input_w},{input_h})->({input_w*self.scaling},{input_h*self.scaling}) but expected ({int(target_w/self.scaling)},{int(target_h/self.scaling)})->({target_w},{target_h}).\n Paths: {input_path}->{target_path}")
        input = self.prepare(input)
        target =  self.prepare(target)
        
        return (input, target)

    def __len__(self):
        return len(self.original_filenames)
    

class CropDataset(data.Dataset):
  def __init__(self, base_dataset, transform, scaling, seed=0, crop=None, cropMode='random', noiseInensity=0):
    super(CropDataset, self).__init__()
    self.base = base_dataset
    self.transform=transform
    self.random = np.random.RandomState(seed)
    self.crop = crop
    self.cropMode = cropMode
    self.scaling = scaling
    self.noise = noiseInensity
    
  def __len__(self):
    return len(self.base)

  def __getitem__(self, idx):
    input, target = self.base[idx]
    if self.crop and self.crop > 10 :
        i, t, crop_info = paralellCrop(input, target, crop=self.crop, scale=self.scaling, random=random, cropMode=self.cropMode)
        input = i
        target = t
        
    if self.noise > 0:
        input += torch.normal(0, self.noise, input.size()) # Add guassian Noise with mean on 0 and sel.noise strength
        
    if self.transform :
        seed = self.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        input = self.transform(input)
        random.seed(seed)  # Force same transform for the target
        torch.manual_seed(seed)  # Force same transform for the target
        target = self.transform(target)
        
    return input, target