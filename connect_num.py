import numpy as np
from scipy.ndimage import label
from PIL import Image
import os
import cv2
from collections import Counter 
path = '/home/fabian/BRL/zhangxin/Codes/SAM2/MAPI/MoCA_test'

# Camouflaged object detection
more_than_one_target = []
for video_name in os.listdir(path):
    list_a = []
    for image in os.listdir(os.path.join(path,video_name)):
        image_path = os.path.join(path, video_name, image)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        image = np.array(image)
        opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        label_image, num_cc = label(opened_image)
        list_a.append(num_cc)

    counts = Counter(list_a) # the most connection_number
    max_count = counts.most_common(1)[0][0]
    if max_count>1:
        print(video_name, ':' ,counts)
        more_than_one_target.append(video_name)
