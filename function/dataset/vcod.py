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
import cv2

class VCOD(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None,
                 variation=0):  # mode='TrainDataset_per_sq'

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode))  # ['image0010', 'image0005',...]

        #########################VOS临时借用，不用即可注释
        # data_path = '/home/fabian/BRL/zhangxin/Datasets/Video_object_segmentation/DAVIS2016'
        # self.name_list = os.listdir(data_path)
        ########################

        # Set the basic information of the dataset
        self.data_path = data_path  # './dataset/vcod'
        self.mode = mode
        self.prompt = prompt  # 'bbox'
        self.img_size = args.image_size  # 1024
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'TrainDataset_per_sq':
            self.video_length = args.video_length  # 2
            print('>>> training with {} videos'.format(len(self.name_list)))
        else:
            self.video_length = None  # 测试的时候置为None
            print('>>> validing with {} videos'.format(len(self.name_list)))

        self.transform_to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        # print('>>> validing with {} videos'.format(len(self.name_list)))


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        # newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, name, 'Imgs')
        mask_path = os.path.join(self.data_path, self.mode, name, 'GT')

        ######################## VOS临时加入使用 2024/11/8
        # img_path = os.path.join(self.data_path, name, 'Imgs')
        # mask_path = os.path.join(self.data_path, name, 'GT')
        #######################

        gt_name = sorted(os.listdir(mask_path))

        data_seg_3d_shape = np.array(Image.open(mask_path + '/' + gt_name[0])).shape  # (512, 512)
        # data_seg_3d_shape = np.load(mask_path + '/00000.png').shape  # (512, 512)
        num_frame = len(os.listdir(mask_path))  # 131
        data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame,))  # shape: (512, 512, 131)

        for i in range(num_frame):
            data_seg_3d[..., i] = np.array(Image.open(os.path.join(mask_path, gt_name[i])))
        for i in range(data_seg_3d.shape[-1]):
            if np.sum(data_seg_3d[..., i]) > 0:  # 如i为45时候,即找到第一个有mask不全为黑的图片
                data_seg_3d = data_seg_3d[..., i:]  # 最后跳出时此shape为(512, 512, 86)
                break
        starting_frame_nonzero = i  # 45
        for j in reversed(range(data_seg_3d.shape[-1])):  # 44,...,0
            if np.sum(data_seg_3d[..., j]) > 0:
                data_seg_3d = data_seg_3d[..., :j + 1]  # :j+1这种索引真正取到的只到j
                break
        num_frame = data_seg_3d.shape[-1]  # 79
        if self.video_length is None:
            # video_length = int(num_frame / 4)  # 原始代码
            video_length = int(num_frame / 1)
            # print("video_length = {}".format(video_length))
        else:
            video_length = self.video_length  # 2
            # print("video_length = {}".format(video_length))
        if num_frame > video_length and self.mode == 'TrainDataset_per_sq':
            starting_frame = np.random.randint(0, num_frame - video_length + 1)
        else:
            starting_frame = 0  # 测试时候为0
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(starting_frame, starting_frame + video_length):
            # img = Image.open(os.path.join(img_path, f'{frame_index + starting_frame_nonzero}.jpg')).convert('RGB')
            image_path = os.path.join(img_path, gt_name[frame_index + starting_frame_nonzero]).replace('.png','.jpg') # str.replace(old, new, count)
            img = Image.open(image_path).convert('RGB')
            mask = data_seg_3d[..., frame_index]
            # mask = np.rot90(mask)
            obj_list = np.unique(mask[mask > 0])  # mask>0满足位置全为True, mask[mask>0]取出为True位置的值变为一维数组, unique把这些值去重复
            diff_obj_mask_dict = {}
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click' or self.prompt == 'central_moment':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                if self.transform_msk:
                    obj_mask = Image.fromarray(obj_mask) # 解决选点的位置要从原始图像大小去选择坐标的问题,这也是VCOD_new里面的
                    # obj_mask = obj_mask.resize(newsize)
                    # obj_mask = self.transform_msk(obj_mask).int()
                    obj_mask = self.transform_to_tensor(obj_mask).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)),
                                                                                         point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation,
                                                            seed=self.seed)  # 返回[y0,x0,y1,x1]
                if self.prompt == 'central_moment':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = central_moment(np.array(obj_mask.squeeze(0)), point_label)

            if self.transform:
                # state = torch.get_rng_state()
                img = self.transform(img)
                # torch.set_rng_state(state)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click' or self.prompt == 'central_moment':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict

        image_meta_dict = {'filename_or_obj': name}
        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
                'file_name_and_shape': (gt_name, data_seg_3d_shape), # 这行新加的,2024/8/15
            }
        elif self.prompt == 'click' or self.prompt == 'central_moment':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
                'file_name_and_shape': (gt_name, data_seg_3d_shape),
            }


class VCOD_new(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None,
                 variation=0):  # mode='TrainDataset_per_sq'

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode))  # ['image0010', 'image0005',...]

        # Set the basic information of the dataset
        self.data_path = data_path  # './dataset/vcod'
        self.mode = mode
        self.prompt = prompt  # 'bbox'
        self.img_size = args.image_size  # 1024
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'TrainDataset_per_sq':
            self.video_length = args.video_length  # 2
        else:
            self.video_length = None  # 测试的时候置为None

        self.transform_to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, name, 'Imgs')
        mask_path = os.path.join(self.data_path, self.mode, name, 'GT')

        gt_name = sorted(os.listdir(mask_path))

        data_seg_3d_shape = np.array(Image.open(mask_path + '/' + gt_name[0])).shape  # (512, 512)
        # data_seg_3d_shape = np.load(mask_path + '/00000.png').shape  # (512, 512)
        num_frame = len(os.listdir(mask_path))  # 131
        data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame,))  # shape: (512, 512, 131)

        for i in range(num_frame):
            data_seg_3d[..., i] = np.array(Image.open(os.path.join(mask_path, gt_name[i])))
        for i in range(data_seg_3d.shape[-1]):
            if np.sum(data_seg_3d[..., i]) > 0:  # 如i为45时候,即找到第一个有mask不全为黑的图片
                data_seg_3d = data_seg_3d[..., i:]  # 最后跳出时此shape为(512, 512, 86)
                break
        starting_frame_nonzero = i  # 45
        for j in reversed(range(data_seg_3d.shape[-1])):  # 44,...,0
            if np.sum(data_seg_3d[..., j]) > 0:
                data_seg_3d = data_seg_3d[..., :j + 1]  # :j+1这种索引真正取到的只到j
                break
        num_frame = data_seg_3d.shape[-1]  # 79
        if self.video_length is None:
            # video_length = int(num_frame / 4)  # 原始代码
            video_length = int(num_frame / 1)
            # print("video_length = {}".format(video_length))
        else:
            video_length = self.video_length  # 2
        if num_frame > video_length and self.mode == 'TrainDataset_per_sq':
            starting_frame = np.random.randint(0, num_frame - video_length + 1)
        else:
            starting_frame = 0  # 测试时候为0
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(starting_frame, starting_frame + video_length):
            # img = Image.open(os.path.join(img_path, f'{frame_index + starting_frame_nonzero}.jpg')).convert('RGB')
            image_path = os.path.join(img_path, gt_name[frame_index + starting_frame_nonzero]).replace('.png','.jpg')
            img = Image.open(image_path).convert('RGB')
            mask = data_seg_3d[..., frame_index]
            # mask = np.rot90(mask)
            obj_list = np.unique(mask[mask > 0])  # mask>0满足位置全为True, mask[mask>0]取出为True位置的值变为一维数组, unique把这些值去重复
            diff_obj_mask_dict = {}
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click' or self.prompt == 'central_moment':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                if self.transform_msk:
                    obj_mask = Image.fromarray(obj_mask)
                    obj_mask = self.transform_to_tensor(obj_mask).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)),
                                                                                         point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation,
                                                            seed=self.seed)  # 返回[y0,x0,y1,x1]
                if self.prompt == 'central_moment':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = central_moment(np.array(obj_mask.squeeze(0)), point_label)

            if self.transform:
                # state = torch.get_rng_state()
                img = self.transform(img)
                # torch.set_rng_state(state)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click' or self.prompt == 'central_moment':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict

        image_meta_dict = {'filename_or_obj': name}
        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
                'file_name_and_shape': (gt_name, data_seg_3d_shape), # 这行新加的,2024/8/15
            }
        elif self.prompt == 'click' or self.prompt == 'central_moment':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
                'file_name_and_shape': (gt_name, data_seg_3d_shape),
            }


'''
2024/9/13  VCOD_EPFlow, 借鉴EPFlow的视频数据加载
'''
class VCOD_Train_EPFlow(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None,
                 variation=0):  # mode='TrainDataset_per_sq'

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode))  # ['image0010', 'image0005',...]

        # Set the basic information of the dataset
        self.data_path = data_path  # './dataset/vcod'
        self.mode = mode
        self.prompt = prompt  # 'bbox'
        self.img_size = args.image_size  # 1024
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'TrainDataset_per_sq':
            # self.video_length = args.video_length  # 2
            self.video_length = None  # 2
        else:
            self.video_length = None  # 测试的时候置为None

        self.transform_to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        ########################
        self.image_pairs_list = []
        self.gts = []
        self.extra_info = []
        ori_root = os.path.join(data_path, mode)
        images_root = ori_root
        for video_name in os.listdir(ori_root):
            image_root = ori_root + '/' + video_name + '/Imgs/'
            gt_root = ori_root + '/'  + video_name + '/GT/'
            self.image = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

            # sorted files
            self.image = sorted(self.image)
            self.gt = sorted(self.gt)[0:-1]

            self.gts += self.gt

            for i in range(len(self.image) - 1):
                if i == 0:
                    self.image_pairs_list += [[self.image[i], self.image[i], self.image[i + 1]]]
                else:
                    self.image_pairs_list += [[self.image[i-1], self.image[i], self.image[i + 1]]]
                frame_name = self.image[i].split('/')[-1].split('.')[0]
                self.extra_info += [(video_name, frame_name)]

            assert len(self.image_pairs_list) == len(self.gts)


        # get size of dataset
        self.size = len(self.image_pairs_list)
        if mode == 'TrainDataset_per_sq':
            # print('>>> trainig/validing with {} samples'.format(self.size))
            print('>>> trainig with {} samples'.format(self.size))
        else:
            print('>>> validing with {} samples'.format(self.size))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # index = 135  # 256
        assert self.image_pairs_list[index][1].split('/')[-1].split('.')[0] == self.gts[index].split('/')[-1].split('.')[0]
        gt_name = self.gts[index].split('/')[-1].split('.')[0]
        name = self.gts[index].split('/')[-3].split('.')[0]
        point_label = 1
        image_pairs = self.image_pairs_list[index]

        ###################临时删除
        # image = cv2.imread(image_pairs[1])
        # image = cv2.resize(image, (1024,1024))
        # path = '/home/fabian/BRL/zhangxin/Codes/SAM2/Medical-SAM2-main/results/00220.jpg'
        # cv2.imwrite(path,image)
        # exit()

        ###################

        data_seg_3d_shape = np.array(Image.open(self.gts[index])).shape  # (720, 1280)
        num_frame = len(self.image_pairs_list[index])  # 3
        data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame,))  # shape: (720, 1280, 3)

        for i in range(num_frame):
            gt_path = self.image_pairs_list[index][i].replace('Imgs','GT')  # (old,new)
            gt_path = gt_path.replace('.jpg','.png')
            data_seg_3d[..., i] = np.array(Image.open(gt_path))

        starting_frame = 0
        img_tensor = torch.zeros(num_frame, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(starting_frame, starting_frame + num_frame):
            img = Image.open(self.image_pairs_list[index][frame_index]).convert('RGB')
            mask = data_seg_3d[..., frame_index]
            # mask = np.rot90(mask)
            obj_list = np.unique(mask[mask > 0])  # mask>0满足位置全为True, mask[mask>0]取出为True位置的值变为一维数组, unique把这些值去重复
            diff_obj_mask_dict = {}
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click' or self.prompt == 'central_moment':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                if self.transform_msk:
                    obj_mask = Image.fromarray(obj_mask) # 解决选点的位置要从原始图像大小去选择坐标的问题,这也是VCOD_new里面的
                    obj_mask = self.transform_to_tensor(obj_mask).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)),
                                                                                         point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation,
                                                            seed=self.seed)  # 返回[y0,x0,y1,x1]
                if self.prompt == 'central_moment':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = central_moment(np.array(obj_mask.squeeze(0)), point_label)

            if self.transform:
                # state = torch.get_rng_state()
                img = self.transform(img)
                # torch.set_rng_state(state)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click' or self.prompt == 'central_moment':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict

        image_meta_dict = {'filename_or_obj': name}
        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
                'file_name_and_shape': (gt_name, data_seg_3d_shape), # 这行新加的,2024/8/15
            }
        elif self.prompt == 'click' or self.prompt == 'central_moment':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
                'file_name_and_shape': (gt_name, data_seg_3d_shape),
            }



class VCOD_Test_EPFlow(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None,
                 variation=0):  # mode='TrainDataset_per_sq'

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode))  # ['image0010', 'image0005',...]

        # Set the basic information of the dataset
        self.data_path = data_path  # './dataset/vcod'
        self.mode = mode
        self.prompt = prompt  # 'bbox'
        self.img_size = args.image_size  # 1024
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        self.transform_to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        ########################
        self.image_pairs_list = []
        self.gts = []
        self.extra_info = []
        ori_root = os.path.join(data_path, mode)
        images_root = ori_root
        for video_name in os.listdir(ori_root):
            image_root = ori_root + '/' + video_name + '/Imgs/'
            gt_root = ori_root + '/'  + video_name + '/GT/'
            self.image = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

            # sorted files
            self.image = sorted(self.image)
            self.gt = sorted(self.gt)[0:-1]

            self.gts += self.gt

            for i in range(len(self.image) - 1):
                if i == 0:
                    self.image_pairs_list += [[self.image[i], self.image[i], self.image[i + 1]]]
                else:
                    self.image_pairs_list += [[self.image[i-1], self.image[i], self.image[i + 1]]]
                frame_name = self.image[i].split('/')[-1].split('.')[0]
                self.extra_info += [(video_name, frame_name)]

            assert len(self.image_pairs_list) == len(self.gts)


        # get size of dataset
        self.size = len(self.image_pairs_list)
        print('>>> validing with {} samples'.format(self.size))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # index = 135  # 256
        assert self.image_pairs_list[index][1].split('/')[-1].split('.')[0] == self.gts[index].split('/')[-1].split('.')[0]
        gt_name = self.gts[index].split('/')[-1].split('.')[0]
        name = self.gts[index].split('/')[-3].split('.')[0]
        point_label = 1

        data_seg_3d_shape = np.array(Image.open(self.gts[index])).shape  # (720, 1280)
        num_frame = len(self.image_pairs_list[index])  # 3
        data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame,))  # shape: (720, 1280, 3)

        for i in range(num_frame):
            gt_path = self.image_pairs_list[index][i].replace('Imgs','GT')  # (old,new)
            gt_path = gt_path.replace('.jpg','.png')
            data_seg_3d[..., i] = np.array(Image.open(gt_path))

        starting_frame = 0
        img_tensor = torch.zeros(num_frame, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(starting_frame, starting_frame + num_frame):
            img = Image.open(self.image_pairs_list[index][frame_index]).convert('RGB')
            mask = data_seg_3d[..., frame_index]
            # mask = np.rot90(mask)
            obj_list = np.unique(mask[mask > 0])  # mask>0满足位置全为True, mask[mask>0]取出为True位置的值变为一维数组, unique把这些值去重复
            diff_obj_mask_dict = {}
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click' or self.prompt == 'central_moment':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                if self.transform_msk:
                    obj_mask = Image.fromarray(obj_mask) # 解决选点的位置要从原始图像大小去选择坐标的问题,这也是VCOD_new里面的
                    obj_mask = self.transform_to_tensor(obj_mask).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)),
                                                                                         point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation,
                                                            seed=self.seed)  # 返回[y0,x0,y1,x1]
                if self.prompt == 'central_moment':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = central_moment(np.array(obj_mask.squeeze(0)), point_label)

            if self.transform:
                # state = torch.get_rng_state()
                img = self.transform(img)
                # torch.set_rng_state(state)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click' or self.prompt == 'central_moment':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict

        image_meta_dict = {'filename_or_obj': name}
        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
                'file_name_and_shape': (gt_name, data_seg_3d_shape), # 这行新加的,2024/8/15
            }
        elif self.prompt == 'click' or self.prompt == 'central_moment':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
                'file_name_and_shape': (gt_name, data_seg_3d_shape),
            }

'''
VCOD all, 2024/9/25, 每个epoch所有图片都训练到， 根据video_length来制作数据集
'''

class VCOD_all_train(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None,
                 variation=0):  # mode='TrainDataset_per_sq'

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode))  # ['image0010', 'image0005',...]

        # Set the basic information of the dataset
        self.data_path = data_path  # './dataset/vcod'
        self.mode = mode
        self.prompt = prompt  # 'bbox'
        self.img_size = args.image_size  # 1024
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'TrainDataset_per_sq':
            self.video_length = args.video_length  # 2
        else:
            self.video_length = None  # 测试的时候置为None

        self.transform_to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.image_pairs_list = []
        self.gts = []

        images_root = os.path.join(data_path, mode)
        for video_name in self.name_list:
            image_root = images_root + '/' + video_name + '/Imgs/'
            gt_root = images_root + '/' + video_name + '/GT/'
            self.image = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

            # sorted files
            self.image = sorted(self.image)
            self.gt = sorted(self.gt)[0:1-self.video_length]

            self.gts += self.gt

            for i in range(len(self.image) - self.video_length + 1):
                image_list = []
                for index in range(self.video_length):
                    image_list.append(self.image[index+i])
                    # image_list.append((self.image[index+i], self.gt[index+i]))

                self.image_pairs_list += [image_list]
                # self.image_pairs_list += [[self.image[i], self.image[i + 1]]]
                # frame_name = self.image[i].split('/')[-1].split('.')[0]
                # self.extra_info += [(video_name, frame_name)]
            assert len(self.image_pairs_list) == len(self.gts)


        self.size = len(self.image_pairs_list)
        print('>>> trainig/validing with {} samples'.format(self.size))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        point_label = 1

        """Get the images"""
        assert self.image_pairs_list[index][0].split('/')[-1].split('.')[0] == self.gts[index].split('/')[-1].split('.')[0]

        data_seg_3d_shape = np.array(Image.open(self.gts[index])).shape  # (512, 512)
        num_frame = len(self.image_pairs_list[index])  # 131
        data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame,))  # shape: (512, 512, 131)

        for i in range(num_frame):
            mask_path_i = self.image_pairs_list[index][i].replace('Imgs', 'GT')
            mask_path_i = mask_path_i.replace('.jpg', '.png')
            data_seg_3d[..., i] = np.array(Image.open(mask_path_i))
        for i in range(data_seg_3d.shape[-1]):
            if np.sum(data_seg_3d[..., i]) > 0:  # 如i为45时候,即找到第一个有mask不全为黑的图片
                data_seg_3d = data_seg_3d[..., i:]  # 最后跳出时此shape为(512, 512, 86)
                break
        starting_frame_nonzero = i  # 45
        for j in reversed(range(data_seg_3d.shape[-1])):  # 44,...,0
            if np.sum(data_seg_3d[..., j]) > 0:
                data_seg_3d = data_seg_3d[..., :j + 1]  # :j+1这种索引真正取到的只到j
                break
        num_frame = data_seg_3d.shape[-1]
        video_length = self.video_length  # 2
        starting_frame = 0  # 测试时候为0
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(starting_frame, starting_frame + video_length):
            image_path = self.image_pairs_list[index][frame_index]
            img = Image.open(image_path).convert('RGB')
            mask = data_seg_3d[..., frame_index]
            # mask = np.rot90(mask)
            obj_list = np.unique(mask[mask > 0])  # mask>0满足位置全为True, mask[mask>0]取出为True位置的值变为一维数组, unique把这些值去重复
            diff_obj_mask_dict = {}
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click' or self.prompt == 'central_moment':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                if self.transform_msk:
                    obj_mask = Image.fromarray(obj_mask)
                    obj_mask = self.transform_to_tensor(obj_mask).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)),
                                                                                         point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation,
                                                            seed=self.seed)  # 返回[y0,x0,y1,x1]
                if self.prompt == 'central_moment':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = central_moment(
                        np.array(obj_mask.squeeze(0)), point_label)

            if self.transform:
                # state = torch.get_rng_state()
                img = self.transform(img)
                # torch.set_rng_state(state)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click' or self.prompt == 'central_moment':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict

        video_name = self.image_pairs_list[index][0].split('/')[-3]
        image_meta_dict = {'filename_or_obj': video_name}
        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
                # 'file_name_and_shape': (gt_name, data_seg_3d_shape),  # 这行新加的,2024/8/15
                'file_name_and_shape': (video_name, data_seg_3d_shape),  # 这行新加的,2024/8/15
            }
        elif self.prompt == 'click' or self.prompt == 'central_moment':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
                'file_name_and_shape': (video_name, data_seg_3d_shape),
            }