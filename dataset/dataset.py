import os
import random

import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 完全移除限制
import torch.utils.data as data
import torchvision.transforms as transforms
from dataset.data_augment import randomRotation, colorEnhance, randomPeper, randomRotation_4


# dataset for training
# class ObjDataset(data.Dataset):
#     def __init__(self, images_root, gts_root, trainsize, dataset='MoCA'):
#         self.trainsize = trainsize
#
#         # get filenames
#         ori_root = images_root
#         self.images = []
#         self.gts = []
#         self.flows = []
#         self.image_pairs_list = []
#         self.extra_info = []
#
#         if dataset == 'MoCA':
#             for video_name in os.listdir(ori_root):
#                 image_root = images_root + video_name + '/Imgs/'
#                 gt_root = gts_root + video_name + '/GT/'
#                 self.image = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
#                 self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
#
#                 # sorted files
#                 self.image = sorted(self.image)
#                 self.gt = sorted(self.gt)[0:-1]
#
#                 self.gts += self.gt
#
#                 for i in range(len(self.image) - 1):
#                     self.image_pairs_list += [[self.image[i], self.image[i + 1]]]
#                     frame_name = self.image[i].split('/')[-1].split('.')[0]
#                     self.extra_info += [(video_name, frame_name)]
#
#                 assert len(self.image_pairs_list) == len(self.gts)
#
#         elif dataset == 'MoCA_pseudo':
#             for video_name in os.listdir(ori_root):
#                 image_root = images_root + video_name + '/Frame/'
#                 gt_root = gts_root + video_name + '/GT/'
#                 self.image = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
#                 self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
#
#                 # sorted files
#                 self.image = sorted(self.image)
#                 self.gt = sorted(self.gt)[0:-1]
#
#                 self.gts += self.gt
#
#                 for i in range(len(self.image) - 1):
#                     self.image_pairs_list += [[self.image[i], self.image[i + 1]]]
#                     frame_name = self.image[i].split('/')[-1].split('.')[0]
#                     self.extra_info += [(video_name, frame_name)]
#
#                 assert len(self.image_pairs_list) == len(self.gts)
#
#         elif dataset == 'VSOD':
#             self.images_all = [f for f in os.listdir(images_root) if f.endswith('.jpg')]
#             self.images = sorted(self.images)
#
#             for idx, image_nm in self.images_all:
#                 if idx==0: continue
#                 if self.images_all[idx].split('_')[0] == self.images_all[idx-1].split('_')[0]:
#                     self.images += [[images_root + self.images_all[idx-1], images_root + self.images_all[idx]]]
#                     self.gts += (gts_root + self.images_all[idx - 1]).replace('.jpg', '.png')
#             # filter mathcing degrees of files
#         # self.filter_files() #
#         # transforms
#         self.img_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#         self.gt_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor()])
#         # get size of dataset
#         self.size = len(self.image_pairs_list)
#         print('>>> trainig/validing with {} samples'.format(self.size))
#
#     def __getitem__(self, index):
#         assert self.image_pairs_list[index][0].split('/')[-1].split('.')[0] == self.gts[index].split('/')[-1].split('.')[0]
#         image1 = self.rgb_loader(self.image_pairs_list[index][0])
#         image2 = self.rgb_loader(self.image_pairs_list[index][1])
#         gt = self.binary_loader(self.gts[index])
#
#         # data augumentation
#         # image1, image2, gt = randomRotation(image1, image2, gt)
#         image1, image2, gt = randomRotation(image1, image2, gt)
#         image1 = colorEnhance(image1)
#         image2 = colorEnhance(image2)
#         gt = randomPeper(gt)
#
#         image1 = self.img_transform(image1)
#         image2 = self.img_transform(image2)
#         gt = self.gt_transform(gt)
#
#         return image1, image2, gt
#
#     def filter_files(self):
#         assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
#         images = []
#         gts = []
#         for img_path, gt_path in zip(self.images, self.gts):
#             img = Image.open(img_path)
#             gt = Image.open(gt_path)
#             if img.size == gt.size:
#                 images.append(img_path)
#                 gts.append(gt_path)
#         self.images = images
#         self.gts = gts
#
#     def rgb_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')
#
#     def binary_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('L')
#
#     def __len__(self):
#         return self.size

from data.utils import collect_r2c_data
import re
#使用正则表达式来判断字符串是否为小数或数字
def is_number_or_decimal(s):
    return bool(re.match(r'^\d+(\.\d+)?$', s))
class Ref_ObjDataset(data.Dataset):
    def __init__(self, images_root, gts_root, trainsize, dataset='MoCA', mode='train'):
        self.trainsize = trainsize

        # get filenames
        if dataset == 'MoCA':
            ori_root = images_root + 'Camo' + '/' + mode + '/'
            self.images = []
            self.gts = []
            self.flows = []
            self.image_pairs_list = []
            self.extra_info = []

            self.mode = mode
            self.ref_file_list = collect_r2c_data(data_root=images_root, mode=self.mode, record_file='./data/refsplits.json')

            path = '/home/fabian/BRL/zhangxin/Datasets/Ref_vcod_v2/Ref/Images/'
            pid2label = {pid: label for label, pid in enumerate(os.listdir(path))}

        elif dataset == 'R2C7K':
            ori_root = images_root + 'Camo' + '/' + mode + '/Imgs/'
            self.images = []
            self.gts = []
            self.flows = []
            self.image_pairs_list = []
            self.extra_info = []

            self.mode = mode
            self.ref_file_list = collect_r2c_data(data_root=images_root, mode=self.mode, record_file='./data/ref_R2C7K_splits.json')

            pid2label = {pid: label for label, pid in enumerate(os.listdir(ori_root))}

        elif dataset == 'COD10K':
            ori_root = images_root
            self.images = []
            self.gts = []
            self.flows = []
            self.image_pairs_list = []
            self.extra_info = []
            self.mode = mode
            pid2label = {pid: label for label, pid in enumerate(os.listdir(ori_root))}

        if dataset == 'MoCA':  # 暂时当ref
            for video_name in os.listdir(ori_root):
                image_root = ori_root + video_name + '/Imgs/'
                gt_root = ori_root + video_name + '/GT/'
                self.image = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
                self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

                # 净化视频名字来选ref, 任意从20个里面选择一个ref
                cat_all = video_name.split('_')
                decision = cat_all[-1]
                if is_number_or_decimal(decision):
                    temp_name = cat_all[:-1]
                    ref_name = '_'.join(temp_name)
                else:
                    ref_name = video_name

                # sorted files
                self.image = sorted(self.image)
                self.gt = sorted(self.gt)[0:-1]

                self.gts += self.gt

                for i in range(len(self.image) - 1): # 视频的
                    # self.image_pairs_list += [[self.image[i], self.image[i + 1]]]
                    # frame_name = self.image[i].split('/')[-1].split('.')[0]
                    # self.extra_info += [(video_name, frame_name)]

                    index = random.randint(0, 19)  # 闭区间,包含0和19，即ref前20张训练
                    # self.image_pairs_list += [[self.image[i], self.image[i + 1], self.ref_file_list[ref_name][index]]] #
                    self.image_pairs_list += [[self.image[i], self.image[i + 1], self.ref_file_list[ref_name][index], pid2label[ref_name]]] # 加label
                    frame_name = self.image[i].split('/')[-1].split('.')[0]
                    self.extra_info += [(video_name, frame_name)]

                assert len(self.image_pairs_list) == len(self.gts)

        elif dataset == 'R2C7K':  # 图片级别的
            for video_name in os.listdir(ori_root):
                image_root = ori_root + video_name + '/'
                gt_root = ori_root + video_name + '/'
                gt_root = gt_root.replace('Imgs','GT')
                self.image = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
                self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

                # 净化视频名字来选ref, 任意从20个里面选择一个ref
                cat_all = video_name.split('_')
                decision = cat_all[-1]
                if is_number_or_decimal(decision):
                    temp_name = cat_all[:-1]
                    ref_name = '_'.join(temp_name)
                else:
                    ref_name = video_name

                # sorted files
                self.image = sorted(self.image)
                self.gt = sorted(self.gt)

                self.gts += self.gt

                for i in range(len(self.image)): # 图片的
                    index = random.randint(0, 19)  # 闭区间,包含0和19，即ref前20张训练
                    # self.image_pairs_list += [[self.image[i], self.image[i + 1], self.ref_file_list[ref_name][index]]] #
                    self.image_pairs_list += [[self.image[i], self.image[i], self.ref_file_list[ref_name][index], pid2label[ref_name]]] # 加label
                    frame_name = self.image[i].split('/')[-1].split('.')[0]
                    self.extra_info += [(video_name, frame_name)]

                assert len(self.image_pairs_list) == len(self.gts)


        elif dataset == 'COD10K':  # 图片级别的
            for video_name in os.listdir(ori_root):
                image_root = ori_root + video_name + '/Imgs/'
                gt_root = ori_root + video_name + '/GT/'
                self.image = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
                self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

                # sorted files
                self.image = sorted(self.image)
                self.gt = sorted(self.gt)
                self.gts += self.gt

                for i in range(len(self.image)): # 图片的
                    self.image_pairs_list += [[self.image[i], self.image[i], self.image[i], 0]] # 加label
                    frame_name = self.image[i].split('/')[-1].split('.')[0]
                    self.extra_info += [(video_name, frame_name)]

                assert len(self.image_pairs_list) == len(self.gts)

        elif dataset == 'MoCA_pseudo':
            for video_name in os.listdir(ori_root):
                image_root = images_root + video_name + '/Frame/'
                gt_root = gts_root + video_name + '/GT/'
                self.image = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
                self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

                # sorted files
                self.image = sorted(self.image)
                self.gt = sorted(self.gt)[0:-1]

                self.gts += self.gt

                for i in range(len(self.image) - 1):
                    self.image_pairs_list += [[self.image[i], self.image[i + 1]]]
                    frame_name = self.image[i].split('/')[-1].split('.')[0]
                    self.extra_info += [(video_name, frame_name)]

                assert len(self.image_pairs_list) == len(self.gts)

        elif dataset == 'VSOD':
            self.images_all = [f for f in os.listdir(images_root) if f.endswith('.jpg')]
            self.images = sorted(self.images)

            for idx, image_nm in self.images_all:
                if idx==0: continue
                if self.images_all[idx].split('_')[0] == self.images_all[idx-1].split('_')[0]:
                    self.images += [[images_root + self.images_all[idx-1], images_root + self.images_all[idx]]]
                    self.gts += (gts_root + self.images_all[idx - 1]).replace('.jpg', '.png')
            # filter mathcing degrees of files
        # self.filter_files() #
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.image_pairs_list)
        print('>>> trainig/validing with {} samples'.format(self.size))

    def __getitem__(self, index):
        assert self.image_pairs_list[index][0].split('/')[-1].split('.')[0] == self.gts[index].split('/')[-1].split('.')[0]
        image1 = self.rgb_loader(self.image_pairs_list[index][0])
        image2 = self.rgb_loader(self.image_pairs_list[index][1])
        image_ref = self.rgb_loader(self.image_pairs_list[index][2])
        gt = self.binary_loader(self.gts[index])

        # data augumentation
        # image1, image2, gt = randomRotation(image1, image2, gt)
        image1, image2, image_ref, gt = randomRotation_4(image1, image2, image_ref, gt)
        image1 = colorEnhance(image1)
        image2 = colorEnhance(image2)
        image_ref = colorEnhance(image_ref)
        gt = randomPeper(gt)

        image1 = self.img_transform(image1)
        image2 = self.img_transform(image2)
        image_ref = self.img_transform(image_ref)
        gt = self.gt_transform(gt)

        label = self.image_pairs_list[index][3]

        return image1, image2, image_ref, gt, label

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

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
    def __init__(self, images_root, gts_root, testsize, dataset_type='MoCA',mode='test'):
        self.testsize = testsize

        if dataset_type == 'MoCA':
            self.gt_list = []
            self.image_pairs_list = []
            self.extra_info = []
            # ori_root = images_root
            self.mode = mode
            self.ref_file_list = collect_r2c_data(data_root=images_root, mode=self.mode, record_file='./data/refsplits.json')
            ori_root = images_root + 'Camo' + '/' + mode + '/'

        elif dataset_type == 'R2C7K':
            self.gt_list = []
            self.image_pairs_list = []
            self.extra_info = []
            # ori_root = images_root
            self.mode = mode
            self.ref_file_list = collect_r2c_data(data_root=images_root, mode=self.mode, record_file='./data/ref_R2C7K_splits.json')
            ori_root = images_root + 'Camo' + '/' + mode + '/Imgs/'

        elif dataset_type == 'COD10K':
            self.gt_list = []
            self.image_pairs_list = []
            self.extra_info = []
            self.mode = mode
            ori_root = images_root


        for video_name in os.listdir(ori_root):
            if 'CAD' in dataset_type:
                image_root = images_root + video_name + '/frames/'
            elif 'pseudo' in dataset_type:
                image_root = images_root + video_name + '/Frame/'
            elif 'R2C7K' in dataset_type:
                image_root = ori_root + video_name + '/'
            else:
                image_root = ori_root + video_name + '/Imgs/'

            if 'R2C7K' in dataset_type:
                gt_root = image_root.replace('Imgs', 'GT')
            else:
                gt_root = ori_root + video_name + '/GT/'
            self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

            self.images = sorted(self.images)
            if 'R2C7K' in dataset_type or 'COD10K' in dataset_type:
                self.gts = sorted(self.gts) # 图片的都要
            else:
                self.gts = sorted(self.gts)[0:-1]
            self.gt_list += self.gts

            # 净化视频名字来选ref, 任意从20个里面选择一个ref
            cat_all = video_name.split('_')
            decision = cat_all[-1]
            if is_number_or_decimal(decision):
                temp_name = cat_all[:-1]
                ref_name = '_'.join(temp_name)
            else:
                ref_name = video_name

            if 'R2C7K' in dataset_type: # 图片的
                for i in range(len(self.images)): # 视频的
                    index = random.randint(0, 4)  # 闭区间,包含0和19
                    self.image_pairs_list += [[self.images[i], self.images[i], self.ref_file_list[ref_name][index]]]
                    frame_name = self.images[i].split('/')[-1].split('.')[0]
                    self.extra_info += [(video_name, frame_name)]

            elif 'COD10K' in dataset_type:
                for i in range(len(self.images)): # 视频的
                    self.image_pairs_list += [[self.images[i], self.images[i], self.images[i]]]
                    frame_name = self.images[i].split('/')[-1].split('.')[0]
                    self.extra_info += [(video_name, frame_name)]

            else:
                for i in range(len(self.images) - 1): # 视频的
                    index = random.randint(0, 4)  # 闭区间,包含0和19
                    self.image_pairs_list += [[self.images[i], self.images[i + 1], self.ref_file_list[ref_name][index]]]
                    frame_name = self.images[i].split('/')[-1].split('.')[0]
                    self.extra_info += [(video_name, frame_name)]

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.gt_transform_2 = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.image_pairs_list)
        self.index = 0
        assert len(self.image_pairs_list) == len(self.gt_list)

    def load_data(self):
        assert self.image_pairs_list[self.index][0].split('/')[-1].split('.')[0] == self.gt_list[self.index].split('/')[-1].split('.')[0]
        image1 = self.rgb_loader(self.image_pairs_list[self.index][0])
        image1 = self.transform(image1).unsqueeze(0)
        image2 = self.rgb_loader(self.image_pairs_list[self.index][1])
        image2 = self.transform(image2).unsqueeze(0)
        image_ref = self.rgb_loader(self.image_pairs_list[self.index][2])
        image_ref = self.transform(image_ref).unsqueeze(0)

        gt = self.binary_loader(self.gt_list[self.index])
        gt_tensor = self.gt_transform_2(gt)

        video_name = self.extra_info[self.index][0]
        name = self.extra_info[self.index][1]

        image_for_post = self.rgb_loader(self.image_pairs_list[self.index][0])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image1, image2, image_ref, gt, gt_tensor, name, video_name, np.array(image_for_post) #

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
    def __init__(self, images_root, testsize, dataset_type='MoCA', mode='test'):
        self.testsize = testsize

        self.gt_list = []
        self.image_pairs_list = []
        self.extra_info = []
        ori_root = images_root + 'Camo' + '/' + mode + '/'
        for video_name in os.listdir(ori_root):
            if 'CAD' in dataset_type:
                image_root = images_root + video_name + '/frames/'
            elif 'pseudo' in dataset_type:
                image_root = images_root + video_name + '/Frame/'
            else:
                image_root = ori_root + video_name + '/Imgs/'
            self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.images = sorted(self.images)

            for i in range(len(self.images) - 1):
                self.image_pairs_list += [[self.images[i], self.images[i + 1]]]
                frame_name = self.images[i].split('/')[-1].split('.')[0]
                self.extra_info += [(video_name, frame_name)]

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.image_pairs_list)
        self.index = 0

    def load_data(self):
        image1 = self.rgb_loader(self.image_pairs_list[self.index][0])
        shape = (image1.height, image1.width)
        image1 = self.transform(image1).unsqueeze(0)
        image2 = self.rgb_loader(self.image_pairs_list[self.index][1])
        image2 = self.transform(image2).unsqueeze(0)

        video_name = self.extra_info[self.index][0]
        name = self.extra_info[self.index][1]

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image1, image2, name, video_name, shape

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