import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2_train.build_sam import build_sam2
from sam2_train.sam2_image_predictor import SAM2ImagePredictor
from func_3d.utils import generate_bbox

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def adjust_bbox(bbox, direction, step=1):
    x1, y1, x2, y2 = bbox

    if direction == 'up':
        y1 = max(y1 - step, 0)  # prevent pop out
    elif direction == 'down':
        y2 += step
    elif direction == 'left':
        x1 = max(x1 - step, 0)
    elif direction == 'right':
        x2 += step

    return np.array([x1, y1, x2, y2])


def adjust_bounding_box(bbox, model, mse_threshold):
    directions = ['up', 'down', 'left', 'right']
    mse_history = {direction: [] for direction in directions}
    fixed_directions = set()

    init_mask, scores, _ = model.predict(
        point_coords=None,
        point_labels=None,
        box=bbox[None, :],
        multimask_output=False,
    )

    while len(fixed_directions) < 4:
        for direction in directions:
            if direction in fixed_directions:
                continue

            new_bbox = adjust_bbox(bbox, direction)
            predicted_mask, scores, _ = model.predict(
                point_coords=None,
                point_labels=None,
                box=new_bbox[None, :],
                multimask_output=False,
            )
            mse_history[direction].append(predicted_mask)


            if len(mse_history[direction]) > 1:
                mse_change = mse(mse_history[direction][-1],mse_history[direction][-2])
                print(new_bbox, mse_change)

                if mse_change > mse_threshold or mse_change<0.00001:
                    fixed_directions.add(direction)
                    print(f"Direction {direction} fixed with MSE change: {mse_change}")

            bbox = new_bbox  # update new_box

    return bbox

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def mse(image1, image2):
    difference = image1 - image2
    squared_difference = np.square(difference)
    mean_squared_difference = np.mean(squared_difference)
    return mean_squared_difference

sam2_checkpoint = "/home/user0/BRL/fabian/BRL/zhangxin/Codes/SAM2/Medical-SAM2-main/checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

import cv2
import random
def central_moment(mask, point_labels=1):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    list_ordinate = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            list_ordinate.append((cx,cy))
            break

    points = np.array(list_ordinate, dtype=np.float32)
    labels = np.array([point_labels for i in range(len(list_ordinate))], np.int32)
    return labels, points

def random_click(mask, point_labels = 1, seed=None):  # seed=None
    # check if all masks are black
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = max_label
    # max agreement position
    indices = np.argwhere(mask == max_label)
    # return point_labels, indices[np.random.randint(len(indices))]
    if seed is not None:
        rand_instance = random.Random(seed)
        rand_num = rand_instance.randint(0, len(indices) - 1)
    else:
        rand_num = random.randint(0, len(indices) - 1)
    output_index_1 = indices[rand_num][0]
    output_index_0 = indices[rand_num][1]
    # return point_labels, np.array([output_index_0, output_index_1])
    # return np.array([point_labels]), np.array([[output_index_0, output_index_1]]) # single points
    return point_labels, [output_index_0, output_index_1]  # multi points no array

def calculate_mask_similarity(mask1, mask2):
    # assert the same size
    assert mask1.shape == mask2.shape, "The masks have different sizes"

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    similarity = intersection.sum() / union.sum()

    return similarity


original_video_path = '/home/user0/BRL/fabian/BRL/zhangxin/Datasets/VCOD'
pvt_pred_path = '/home/user0/BRL/fabian/BRL/zhangxin/Codes/SAM2/Model/output1'
dataset_name = 'MoCA_test'  # 'CAD_eval'


all_video_name = sorted(os.listdir(os.path.join(original_video_path, dataset_name)))
pred_path = sorted(os.listdir(os.path.join(pvt_pred_path, dataset_name)))


dict_point_content = {}

import time
start_time = time.time()
for video_name in all_video_name:
    pred_path = os.path.join(pvt_pred_path, dataset_name, video_name)
    preds = sorted(os.listdir(pred_path))
    loss_list = []
    target_list = []
    for index, name in enumerate(preds):
        pvt_pred = Image.open(os.path.join(pred_path, name))
        pvt_pred = np.array(pvt_pred.convert("L"))

        point_number = 5
        input_label_list, input_point_list = [],[]
        for i in range(point_number):
            input_label, input_point = random_click(pvt_pred)
            input_label_list.append(input_label)
            input_point_list.append(input_point)

        input_point = np.array(input_point_list)
        input_label = np.array(input_label_list)
        ############################


        if dataset_name == 'MoCA_test':
            image_path = os.path.join(original_video_path, dataset_name, video_name, 'Imgs', name.replace('.png', '.jpg')) # MoCA_test
        elif dataset_name == 'CAD_eval':
            image_path = os.path.join(original_video_path, dataset_name, video_name, 'frames', name.replace('.png', '.jpg')) # CAD_eval
        elif dataset_name == 'DAVIS2016' or dataset_name == 'FBMS_Testset' or dataset_name == 'SegTrackv2' or dataset_name == 'ViSal':
            image_path = os.path.join(original_video_path, dataset_name, video_name, 'Imgs', name.replace('.png', '.jpg'))
        else:
            print("dataset name wrong")
            exit()
        image_sam = Image.open(image_path)
        image_sam = np.array(image_sam.convert("RGB"))
        predictor.set_image(image_sam)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        sorted_ind = np.argsort(scores)[::-1]
        mask = masks[sorted_ind][0]

        loss = calculate_mask_similarity(pvt_pred, mask)
        loss_list.append(loss)
        target_list.append((image_path, input_point))



    max_value = max(loss_list)
    max_index = loss_list.index(max_value)
    arr = np.array(loss_list)
    Top_k = 3
    indices = np.argsort(arr)
    top_n_indices = indices[-Top_k:]
    print(f"前{Top_k}个最大值的索引是: {top_n_indices}")
    top_n_indices = top_n_indices[::-1]

    top_k_content = []
    for i in range(Top_k):
        print(top_n_indices)
        index = top_n_indices[i]
        image_temp_path = target_list[index][0]
        if dataset_name == 'MoCA_test':
            image_temp_path = image_temp_path.replace('Imgs', 'GT')  # MoCA
        elif dataset_name == 'CAD_eval':
            image_temp_path = image_temp_path.replace('frames','GT') # CAD
        else:
            image_temp_path = image_temp_path.replace('Imgs', 'GT')  # VOS
        image_temp_path = image_temp_path.replace('.jpg','.png')
        image_temp = Image.open(image_temp_path)
        pixel_value_list = []
        for i in range(len(target_list[index][1])):
            pixel_value = image_temp.getpixel(target_list[index][1][i])
            pixel_value_list.append(pixel_value)


        temp_list = []
        temp_list.append(arr[index])
        temp_list.append(int(index)) # str)
        temp_list.append(target_list[index][0])  # image_path
        temp_list.append(target_list[index][1].tolist())  # input_point
        image_path = target_list[index][0]
        image_sam = Image.open(image_path)
        image_sam = np.array(image_sam.convert("RGB"))
        predictor.set_image(image_sam)
        input_point = target_list[index][1]
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        ) # masks会预测出3张
        input_bbox = generate_bbox(mask)
        final_bbox = adjust_bounding_box(input_bbox, predictor, 0.0005)
        temp_list.append(pixel_value_list)
        temp_list.append(final_bbox.tolist())
        top_k_content.append(temp_list)
    dict_point_content[video_name] = top_k_content


    with open(os.path.join('/home/user0/BRL/fabian/BRL/zhangxin/Codes/SAM2/Medical-SAM2-main/gmflow', 'point.txt'), 'a') as file:
        file.write('{};{};{};{}\n'.format(max_value, max_index, target_list[max_index], pixel_value_list))
    print('{} finished.'.format(video_name))

# 将字典存储到JSON文件
import json
with open('/home/user0/BRL/fabian/BRL/zhangxin/Codes/SAM2/Medical-SAM2-main/gmflow/data.json', 'w') as json_file:
    json.dump(dict_point_content, json_file)
exit()