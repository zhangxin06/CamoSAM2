""" Dataloader for the VCOD dataset
    Xin Zhang
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from func_3d.utils import random_click, generate_bbox, central_moment
import torchvision.transforms as transforms

class COD(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None,
                 variation=0):  # mode='TrainDataset_per_sq'

        # Set the data list for training
        if mode == 'TrainDataset':
            image_root = os.path.join(data_path, mode, 'Imgs')
            gt_root = os.path.join(data_path, mode, 'GT')
        else:
            test_data_name = 'CAMO'
            image_root = os.path.join(data_path, mode, test_data_name, 'Imgs')
            gt_root = os.path.join(data_path, mode, test_data_name, 'GT')

        self.image = [image_root + '/' + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gt = [gt_root + '/' + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

        # sorted files
        self.image = sorted(self.image)
        self.gt = sorted(self.gt)

        # self.name_list = os.listdir(os.path.join(data_path, mode))  # ['image0010', 'image0005',...]

        # Set the basic information of the dataset
        self.data_path = data_path  # './dataset/vcod'
        self.mode = mode
        self.prompt = prompt  # 'bbox'
        self.img_size = args.image_size  # 1024
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        # if mode == 'TrainDataset_per_sq':
        #     self.video_length = args.video_length  # 2
        # else:
        #     self.video_length = None  # 测试的时候置为None

        self.transform_to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.image)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __getitem__(self, index):
        """Get the images"""
        image = self.rgb_loader(self.image[index])
        gt = self.binary_loader(self.gt[index])
        gt_name = self.gt[index].split('/')[-1]

        # data augumentation
        # image, gt = randomRotation(image, gt)
        # image = colorEnhance(image)
        # gt = randomPeper(gt)  # EPFlow里面给gt加椒盐噪声

        gt_dict = {}
        gt_dict_1 = {}
        mask = np.array(gt)
        obj_list = np.unique(mask[mask>0])
        for obj in obj_list:
            obj_mask = mask == obj
            obj_mask = self.transform_to_tensor(obj_mask).int()
            gt_dict_1[obj] = obj_mask

        gt_dict[0] = gt_dict_1
        image = self.transform(image)
        # gt = self.transform_to_tensor(gt).int()
        data_seg_3d_shape = mask.shape[-2:]

        image_meta_dict = {'filename_or_obj': gt_name}

        return {
            'image': image,
            'label': gt_dict,
            'bbox': mask,
            'image_meta_dict': image_meta_dict,
            'file_name_and_shape': (gt_name, data_seg_3d_shape), # 这行新加的,2024/8/15
        }


import numpy as np
from PIL import Image, ImageEnhance

import torchvision.transforms as transforms
import random

def randomRotation(img, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        img = img.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return img, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)
