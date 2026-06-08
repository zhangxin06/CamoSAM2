""" Dataloader for the BTCV dataset
    Yunli Qi
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from func_3d.utils import random_click, generate_bbox


class BTCV(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', seed=None, variation=0):

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode, 'image'))  # ['image0010', 'image0005',...]
        
        # Set the basic information of the dataset
        self.data_path = data_path  # './dataset/btcv'
        self.mode = mode
        self.prompt = prompt  # 'bbox'
        self.img_size = args.image_size  # 1024
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'Training':
            self.video_length = args.video_length  # 2
        else:
            self.video_length = None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'image', name)  # './dataset/btcv/Training/image/img0026'
        mask_path = os.path.join(self.data_path, self.mode, 'mask', name)  # './dataset/btcv/Training/mask/img0026'
        data_seg_3d_shape = np.load(mask_path + '/0.npy').shape  # (512, 512)
        num_frame = len(os.listdir(mask_path))  # 131
        data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame,)) # shape: (512, 512, 131)
        for i in range(num_frame):
            data_seg_3d[..., i] = np.load(os.path.join(mask_path, f'{i}.npy'))
        for i in range(data_seg_3d.shape[-1]):
            if np.sum(data_seg_3d[..., i]) > 0: # 如i为45时候,即找到第一个有mask不全为黑的图片
                data_seg_3d = data_seg_3d[..., i:] # 最后跳出时此shape为(512, 512, 86)
                break
        starting_frame_nonzero = i  # 45
        for j in reversed(range(data_seg_3d.shape[-1])): # 44,...,0
            if np.sum(data_seg_3d[..., j]) > 0:
                data_seg_3d = data_seg_3d[..., :j+1] # :j+1这种索引真正取到的只到j
                break
        num_frame = data_seg_3d.shape[-1] # 79
        if self.video_length is None:
            video_length = int(num_frame / 4)
        else:
            video_length = self.video_length  # 2
        if num_frame > video_length and self.mode == 'Training':
            starting_frame = np.random.randint(0, num_frame - video_length + 1)
        else:
            starting_frame = 0
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(starting_frame, starting_frame + video_length):
            img = Image.open(os.path.join(img_path, f'{frame_index + starting_frame_nonzero}.jpg')).convert('RGB')
            mask = data_seg_3d[..., frame_index]
            # mask = np.rot90(mask)
            obj_list = np.unique(mask[mask > 0]) # mask>0满足位置全为True, mask[mask>0]取出为True位置的值变为一维数组, unique把这些值去重复
            diff_obj_mask_dict = {}
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                if self.transform_msk:
                    obj_mask = Image.fromarray(obj_mask)
                    obj_mask = obj_mask.resize(newsize)
                    obj_mask = self.transform_msk(obj_mask).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed) # 返回[y0,x0,y1,x1]
            if self.transform:
                # state = torch.get_rng_state()
                img = self.transform(img)
                # torch.set_rng_state(state)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict


        image_meta_dict = {'filename_or_obj':name}
        if self.prompt == 'bbox':
            return {
                'image':img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict':image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image':img_tensor,
                'label': mask_dict,
                'p_label':point_label_dict,
                'pt':pt_dict,
                'image_meta_dict':image_meta_dict,
            }