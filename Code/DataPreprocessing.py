import os
import random

import cv2
import numpy as np
import pandas as pd


def image_feature_extraction(samples, img_dir, image_size):
    dataset = []
    for i in range(len(samples)):
        data = samples.iloc[i]
        image_path = data[5].replace("'", "")
        image_file_path = os.path.join(img_dir, image_path)
        try:
            image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (image_size, image_size))
        except:
            continue
        dataset.append([np.array(image), np.array(data[4])])
    random.shuffle(dataset)
    return dataset


def uploaded_image_dataset_feature_extraction(img_dir, image_size):
    dataset = []
    image_path = os.path.join(img_dir)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (image_size, image_size))
    dataset.append([np.array(image), np.array([])])
    random.shuffle(dataset)
    return dataset


