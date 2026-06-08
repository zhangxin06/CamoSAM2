import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.losses import DiceLoss, FocalLoss
from tqdm import tqdm

import cfg
from conf import settings
from function.utils import eval_seg
import numpy as np
from PIL import Image

args = cfg.parse_args()
GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(1,7))

torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

import torchvision.transforms as transforms
transform_gt_compare = transforms.Compose([
    transforms.Resize((384, 384)),
])

transform_train_flow = transforms.Compose([
    transforms.Resize((352, 352)),
])



def test_sam(args, val_loader, net: nn.Module, save_dataset_name, json_data):
    net.eval()

    n_val = len(val_loader)  # the number of batch
    mix_res = (0,) * 1 * 2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    prompt_freq = args.prompt_freq

    prompt = args.prompt


    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for pack in val_loader:
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)
            frame_id = list(range(imgs_tensor.size(0)))

            video_height = pack['file_name_and_shape'][1][0]
            video_width = pack['file_name_and_shape'][1][1]

            train_state = net.val_init_state(imgs_tensor=imgs_tensor, video_height=video_height,
                                             video_width=video_width)

            # train_state = net.val_init_state(imgs_tensor=imgs_tensor)
            prompt_freq = len(frame_id)  # means prompt only one
            prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
            obj_list = []
            for id in frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))  # the numbers to be tracked
            if len(obj_list) == 0:
                continue

            video_name = pack['image_meta_dict']['filename_or_obj'][0]
            gt_name = pack['file_name_and_shape'][0]

            with torch.no_grad():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click': # here, click means point+rbox
                                top_k = len(json_data[video_name])
                                for index in range(top_k):
                                    # id = json_data[video_name][index][1]
                                    # points = json_data[video_name][index][-2]
                                    # points = torch.tensor(points).cuda()
                                    # print(json_data[video_name][index][2].split('/')[-1], gt_name[id][0])

                                    # labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                    # _, _, _ = net.train_add_new_points(
                                    #     inference_state=train_state,
                                    #     frame_idx=id,
                                    #     obj_id=ann_obj_id,
                                    #     points=points,
                                    #     labels=labels,
                                    #     clear_old_points=False,
                                    # )
                                    ################################################### add box
                                    id = json_data[video_name][index][1]
                                    points = json_data[video_name][index][-3]
                                    points = torch.tensor(points).cuda()
                                    point_number = len(points)
                                    labels = []
                                    for i in range(point_number):
                                        labels.append(1)
                                    labels = torch.tensor(labels).to(device=GPUdevice)
                                    bbox = json_data[video_name][index][-1]
                                    bbox = torch.tensor(bbox)
                                    print(json_data[video_name][index][2].split('/')[-1], gt_name[id][0])
                                    # labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                    _, _, _ = net.train_add_new_points(
                                        inference_state=train_state,
                                        frame_idx=id,
                                        obj_id=ann_obj_id,
                                        points=points,
                                        labels=labels,
                                        clear_old_points=False,
                                    )
                                    _, _, _ = net.train_add_new_bbox(
                                        inference_state=train_state,
                                        frame_idx=id,
                                        obj_id=ann_obj_id,
                                        bbox=bbox.to(device=GPUdevice),
                                        clear_old_points=False,
                                    )
                                    ###################################################

                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _ = net.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask=torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results

                for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state,
                                                                                          start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }


                for id in frame_id:  # all id
                    for ann_obj_id in obj_list:  # according target
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        map_save_path_final = f'./results/{save_dataset_name}/{video_name}/'
                        os.makedirs(map_save_path_final, exist_ok=True)
                        Image.fromarray(np.uint8((pred[0, 0, :, :].cpu().numpy() > 0.5) * 255)).convert('L').save(map_save_path_final + pack['file_name_and_shape'][0][id][0])
                        print('>>> prediction save at: {}'.format(map_save_path_final + pack['file_name_and_shape'][0][id][0]))

            pbar.update()
