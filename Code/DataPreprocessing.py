import os
import random

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def image_feature_extraction(samples, img_dir, image_size):
    dataset = []
    for i in range(len(samples)):
        data = samples.iloc[i]
        image_path = data[5].replace("'", "")
        image_file_path = os.path.join(img_dir, image_path)
        try:
            y1 = data[1]
            y2 = data[3]
            x1 = data[0]
            x2 = data[2]
            image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
            cropped_image = image[y1:y2, x1:x2]
            image_resized = cv2.resize(cropped_image, (image_size, image_size))
            dataset.append([np.array(image_resized), np.array(data[4])])
        except:
            continue

    random.shuffle(dataset)
    return dataset
