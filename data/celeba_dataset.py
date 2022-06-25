# -*- coding:utf-8 -*-
import os
import random
import numpy as np
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as tf

from data.base_dataset import BaseDataset


class CelebADataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.shuffle = True if opt.isTrain else False 
        self.lr_size = opt.load_size // opt.scale_factor  # 128
        self.hr_size = opt.load_size  # 16

        self.img_dir = opt.dataroot  # 数据集路径
        self.img_names = self.get_img_names()   # 获取图片的名字
        # 数据增强 -- 随机水平翻转
        self.aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                Scale((1.0, 1.3), opt.load_size) 
                ])
        # ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，
        # 其将每一个数值归一化到[0,1]
        # Normalize 则其作用就是先将输入归一化到(0,1)，再使用公式"(x-mean)/std"，将每个元素分布到(-1,1)
        self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    # 获取图片的名字
    def get_img_names(self,):
        img_names = [x for x in os.listdir(self.img_dir)] 
        if self.shuffle:
            random.shuffle(img_names)
        return img_names


    def __len__(self,):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        # 使用Image.open读出图像，加convert('RGB')的作用。
        # 读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，
        # 因此使用convert(‘RGB’)进行通道转换
        # 补充：convert的转化模式有九种不同模式: 1，L，P，RGB，RGBA，CMYK，YCbCr，I，F。
        hr_img = Image.open(img_path).convert('RGB')
        hr_img = self.aug(hr_img)  # 数据增强

        # downsample and upsample to get the LR image
        lr_img = hr_img.resize((self.lr_size, self.lr_size), Image.BICUBIC)  # 16*16
        lr_img_up = lr_img.resize((self.hr_size, self.hr_size), Image.BICUBIC)  # 128*128

        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img_up)

        return {'HR': hr_tensor, 'LR': lr_tensor, 'HR_paths': img_path}


class Scale():
    """
    Random scale the image and pad to the same size if needed.
    ---------------
    # Args:
        factor: tuple input, max and min scale factor.        
    """
    def __init__(self, factor, size):
        self.factor = factor 
        rc_scale = (2 - factor[1], 1)
        self.size   = (size, size)
        self.rc_scale = rc_scale
        self.ratio = (3. / 4., 4. / 3.) 
        self.resize_crop = transforms.RandomResizedCrop(size, rc_scale)

    def __call__(self, img):
        scale_factor = random.random() * (self.factor[1] - self.factor[0]) + self.factor[0]  
        w, h = img.size
        sw, sh = int(w*scale_factor), int(h*scale_factor)
        scaled_img = tf.resize(img, (sh, sw))
        if sw > w:
            i, j, h, w = self.resize_crop.get_params(img, self.rc_scale, self.ratio)
            scaled_img = tf.resized_crop(img, i, j, h, w, self.size, Image.BICUBIC) 
        elif sw < w:
            lp = (w - sw) // 2
            tp = (h - sh) // 2 
            padding = (lp, tp, w - sw - lp, h - sh - tp) 
            scaled_img = tf.pad(scaled_img, padding)
        return scaled_img 

