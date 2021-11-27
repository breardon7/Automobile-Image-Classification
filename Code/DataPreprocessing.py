import os
import random
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

IMAGE_SIZE = 224

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical', input_shape=(IMAGE_SIZE,
                                                                       IMAGE_SIZE,
                                                                       3)),
    tf.keras.layers.RandomRotation(0.2)
])


def image_feature_extraction(samples, img_dir, image_size, augment=False):
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
            if augment:
                image_resized = data_augmentation(tf.expand_dims(image_resized, 0))[0]
                if i < 5:
                    img = Image.fromarray(cropped_image, 'RGB')
                    img.show()
            dataset.append([np.array(image_resized), np.array(data[4])])
        except:
            continue

    random.shuffle(dataset)
    return dataset
