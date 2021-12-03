import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import BackgroundRemoval

IMAGE_SIZE = 224
CHANNELS = 3

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical', input_shape=(IMAGE_SIZE,
                                                                       IMAGE_SIZE,
                                                                       CHANNELS)),
    tf.keras.layers.RandomRotation(0.2)
])


def image_feature_extraction(samples, img_dir):
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
            background_removed_image = BackgroundRemoval.remove_image_background(cropped_image)
            image_resized = cv2.resize(background_removed_image, (IMAGE_SIZE, IMAGE_SIZE))
            image_normalized = cv2.normalize(image_resized, None, alpha=0, beta=1,
                                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # normalize image
            dataset.append([np.array(image_normalized), np.array(data[4])])
        except:
            continue
    random.shuffle(dataset)
    return dataset


def augment_data(samples):
    augmented_dataset = []
    for image in samples:
        image_array = image[0]
        augmented_image = data_augmentation(tf.expand_dims(image_array, 0))[0]
        augmented_dataset.append([np.array(augmented_image), np.array(image[1])])
    random.shuffle(augmented_dataset)
    return augmented_dataset
