import os
import random
import cv2
import numpy as np
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
            image_normalized = cv2.normalize(image_resized, None, alpha=0, beta=1,
                                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # normalize image
            if i < 2:
                cv2.imwrite('cropped_vehicle.png', image_normalized)
            if augment:
                aug_image = data_augmentation(tf.expand_dims(image_normalized, 0))[0]
                dataset.append([np.array(aug_image), np.array(data[4])])
            else:
                dataset.append([np.array(image_normalized), np.array(data[4])])
        except:
            continue

    random.shuffle(dataset)
    return dataset

