import os
import random

import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 完全移除限制
import torch.utils.data as data
import torchvision.transforms as transforms
from dataset.data_augment import randomRotation, colorEnhance, randomPeper, randomRotation_4

from data.utils import collect_r2c_data
import re
#使用正则表达式来判断字符串是否为小数或数字
def is_number_or_decimal(s):
    return bool(re.match(r'^\d+(\.\d+)?$', s))
class Ref_ObjDataset(data.Dataset):
    def __init__(self, images_root, gts_root, trainsize, dataset='MoCA', mode='train', clip_len=5):
        self.trainsize = trainsize

        # get filenames
        if dataset == 'MoCA':
            ori_root = images_root
            self.images = []
            self.gts = []
            self.clip_ref_pairs_list = []
            self.extra_info = []
            self.mode = mode


            for video_name in os.listdir(ori_root):
                vid_path = ori_root + video_name + '/Imgs/'
                frms = sorted(os.listdir(vid_path))
                self.image = []
                self.gt = []
                for idx in range(len(frms)):
                    clip = []
                    for ii in range(-clip_len // 2 + 1, clip_len // 2 + 1):
                        # pick_idx = idx + ii if idx - ii < 0 else idx - ii
                        pick_idx = idx + ii
                        if pick_idx >= len(frms):
                            pick_idx = len(frms) - 1
                        if pick_idx < 0:
                            pick_idx = 0
                        clip.append(os.path.join(vid_path, frms[pick_idx]))
                    self.image.append(clip)
                    # self.gts.append([x.replace("Frame", "GT").replace("jpg", "png") for x in clip])
                    self.gt.append([x.replace("Imgs", "GT").replace("jpg", "png") for x in clip])

                # sorted files
                self.image = sorted(self.image)
                self.gt = sorted(self.gt)

                self.gts += self.gt

                # 净化视频名字来选ref, 任意从20个里面选择一个ref
                cat_all = video_name.split('_')
                decision = cat_all[-1]
                if is_number_or_decimal(decision):
                    temp_name = cat_all[:-1]
                    ref_name = '_'.join(temp_name)
                else:
                    ref_name = video_name

                for i in range(len(self.image)): # 视频的
                    # self.image_pairs_list += [[self.image[i], self.image[i + 1], self.ref_file_list[ref_name][index], pid2label[ref_name]]] # 加label
                    self.clip_ref_pairs_list += [[self.image[i]]] # 加label
                    # frame_name = self.image[i].split('/')[-1].split('.')[0]
                    # self.extra_info += [(video_name, frame_name)]

            assert len(self.clip_ref_pairs_list) == len(self.gts)


        self.size = len(self.clip_ref_pairs_list)
        print(self.size)
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        # self.size = len(self.image_pairs_list)
        print('>>> trainig/validing with {} samples'.format(self.size))

    def __getitem__(self, index):
        assert self.clip_ref_pairs_list[index][0][2].split('/')[-1].split('.')[0] == self.gts[index][2].split('/')[-1].split('.')[0]
        images = [self.rgb_loader(x) for x in self.clip_ref_pairs_list[index][0]]
        gt = [self.binary_loader(x) for x in self.gts[index]]

        # 这里有个创新点, 可以使用edge损失添油加醋, 2024/7/15,后面再考虑加不加

        # data augumentation
        # image1, image2, gt = randomRotation(image1, image2, gt) # 随机反转的数据增强，看后面要否
        # 下面都是各种数据增强, 提点的话可以考虑在这里下手
        images = [self.img_transform(colorEnhance(x)) for x in images]

        for i in range(len(gt)):
            gt[i] = np.array(gt[i])
            gt[i] = Image.fromarray(gt[i])
            gt[i] = self.gt_transform(randomPeper(gt[i]))

        return torch.stack(images), torch.stack(gt)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):  # 这里决定的了index的最大取值
        return self.size

# dataloader for training
def get_loader(image_root, gt_root, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True, multi_gpu=False, dataset_type='MoCA', mode='train'):
    # dataset = ObjDataset(image_root, gt_root, trainsize, dataset_type)
    dataset = Ref_ObjDataset(image_root, gt_root, trainsize, dataset_type, mode)
    print(dataset.__len__)
    if multi_gpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=True
        )
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batchsize,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory,
                                      sampler=train_sampler)
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batchsize,
                                      shuffle=shuffle,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, images_root, gts_root, testsize, dataset_type='MoCA',mode='test', clip_len=5):
        self.testsize = testsize

        if dataset_type == 'MoCA':
            ori_root = images_root
            self.images = []
            self.gts = []
            self.clip_ref_pairs_list = []
            self.extra_info = []
            self.mode = mode

        for video_name in os.listdir(ori_root):
            vid_path = ori_root + video_name + '/Imgs/'
            frms = sorted(os.listdir(vid_path))
            self.image = []
            self.gt = []
            for idx in range(len(frms)):
                clip = []
                for ii in range(-clip_len // 2 + 1, clip_len // 2 + 1):
                    # pick_idx = idx + ii if idx - ii < 0 else idx - ii
                    pick_idx = idx + ii
                    if pick_idx >= len(frms):
                        pick_idx = len(frms) - 1
                    if pick_idx < 0:
                        pick_idx = 0
                    clip.append(os.path.join(vid_path, frms[pick_idx]))
                self.image.append(clip)
                # self.gts.append([x.replace("Frame", "GT").replace("jpg", "png") for x in clip])
                self.gt.append([x.replace("Imgs", "GT").replace("jpg", "png") for x in clip])

            # sorted files
            self.image = sorted(self.image)
            self.gt = sorted(self.gt)

            self.gts += self.gt

            # 净化视频名字来选ref, 任意从20个里面选择一个ref
            cat_all = video_name.split('_')
            decision = cat_all[-1]
            if is_number_or_decimal(decision):
                temp_name = cat_all[:-1]
                ref_name = '_'.join(temp_name)
            else:
                ref_name = video_name

            for i in range(len(self.image)): # 视频的
                index = random.randint(0, 4)  # 闭区间,包含0和19
                self.clip_ref_pairs_list += [
                    [self.image[i]]]
        assert len(self.clip_ref_pairs_list) == len(self.gts)

        self.size = len(self.clip_ref_pairs_list)
        print(self.size)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.gt_transform_2 = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.index = 0
        print('>>> Testing with {} samples'.format(self.size))

    def load_data(self):
        assert self.clip_ref_pairs_list[self.index][0][2].split('/')[-1].split('.')[0] == \
               self.gts[self.index][2].split('/')[-1].split('.')[0]
        images = [self.rgb_loader(x) for x in self.clip_ref_pairs_list[self.index][0]]
        gt = [self.binary_loader(x) for x in self.gts[self.index]]

        images = [self.transform(x) for x in images]
        gt_tensor = [self.gt_transform_2(x) for x in gt]

        self.index += 1
        self.index = self.index % self.size

        return torch.stack(images).unsqueeze(0),  gt, torch.stack(gt_tensor).unsqueeze(0) # 这里加.unsqueeze(0),主要把batch_size维度体现出来为1

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size



class eval_dataset:
    def __init__(self, images_root, testsize, dataset_type='MoCA', mode='test', clip_len=5):
        self.testsize = testsize

        ori_root = images_root
        self.images = []
        self.gts = []
        self.clip_ref_pairs_list = []
        self.extra_info = []
        self.mode = mode

        for video_name in os.listdir(ori_root):
            original_video_name = video_name
            vid_path = ori_root + video_name + '/Imgs/'
            frms = sorted(os.listdir(vid_path))
            self.image = []
            self.gt = []
            for idx in range(len(frms)):
                clip = []
                for ii in range(-clip_len // 2 + 1, clip_len // 2 + 1):
                    # pick_idx = idx + ii if idx - ii < 0 else idx - ii
                    pick_idx = idx + ii
                    if pick_idx >= len(frms):
                        pick_idx = len(frms) - 1
                    if pick_idx < 0:
                        pick_idx = 0
                    clip.append(os.path.join(vid_path, frms[pick_idx]))
                self.image.append(clip)
                # self.gts.append([x.replace("Frame", "GT").replace("jpg", "png") for x in clip])
                self.gt.append([x.replace("Imgs", "GT").replace("jpg", "png") for x in clip])

            # sorted files
            self.image = sorted(self.image)
            self.gt = sorted(self.gt)

            self.gts += self.gt

            # 净化视频名字来选ref, 任意从20个里面选择一个ref
            cat_all = video_name.split('_')
            decision = cat_all[-1]
            if is_number_or_decimal(decision):
                temp_name = cat_all[:-1]
                ref_name = '_'.join(temp_name)
            else:
                ref_name = video_name

            for i in range(len(self.image)):  # 视频的
                self.clip_ref_pairs_list += [
                    [self.image[i]]]
                frame_name = self.image[i][clip_len//2].split('/')[-1].split('.')[0]
                self.extra_info += [(original_video_name, frame_name)]

        assert len(self.clip_ref_pairs_list) == len(self.gts)

        self.size = len(self.clip_ref_pairs_list)
        print(self.size)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.gt_transform_2 = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.index = 0
        print('>>> Testing with {} samples'.format(self.size))

    def load_data(self):
        assert self.clip_ref_pairs_list[self.index][0][2].split('/')[-1].split('.')[0] == \
               self.gts[self.index][2].split('/')[-1].split('.')[0]
        images = [self.rgb_loader(x) for x in self.clip_ref_pairs_list[self.index][0]]
        gt = [self.binary_loader(x) for x in self.gts[self.index]]

        images = [self.transform(x) for x in images]
        gt_tensor = [self.gt_transform_2(x) for x in gt]

        video_name = self.extra_info[self.index][0]
        frame_name = self.extra_info[self.index][1]

        self.index += 1
        self.index = self.index % self.size

        return torch.stack(images).unsqueeze(0), gt, torch.stack(gt_tensor).unsqueeze(0), frame_name, video_name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size