import itertools
import os
import warnings
from random import random
from sys import platform

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.utils.np_utils import to_categorical

from Code import DataPreprocessing, BackgroundRemoval
from Code.ModelMetrics import model_predict

warnings.filterwarnings('ignore')
# i added this security line for mac users because the keychain access blocks unverified ssl from the dataset url
if platform == "darwin":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

IMAGE_SIZE = 224
CHANNELS = 3
BATCH_SIZE = 64
MONITOR_VAL = "val_accuracy"
DROPOUT = 0.5
LOSS = "categorical_crossentropy"
EPOCHS = 3
CLASSES = 197


def model_definition_and_training(pretrained, custom):
    # Hyper Parameters

    # Train Dataset creation
    train_images = np.load('DataStorage/train_data.npy', allow_pickle=True)
    x_train = np.array([i[0] for i in train_images]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    y_train = np.array([i[1] for i in train_images])
    print("TRAIN AND AUGMENTATION DATA PREPROCESSING COMPLETED..")

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    print("TEST DATA PREPROCESSING COMPLETED..")

    if pretrained:
        # // pretrained model - vgg19
        vgg = VGG19(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
                    classes=CLASSES)
        for layer in vgg.layers:
            layer.trainable = False
        model = Sequential()
        model.add(vgg)
        model.add(Flatten())
        model.add(Dense(CLASSES, activation="softmax"))
        checkpoint = ModelCheckpoint("SavedModel/pretrained_vgg19.h5", monitor=MONITOR_VAL, verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False, save_freq=1)
        model.compile(optimizer='Adam', loss=LOSS, metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor=MONITOR_VAL, patience=5, verbose=1)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val),
                  verbose=1, callbacks=[checkpoint, early_stopping])
        model_predict(model)
        model.save("SavedModel/pretrained_vgg19.h5")


def image_feature_extraction(samples, img_dir):
    dataset = []
    image_count = 1
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
            print("Processed Image " + str(image_count))
            image_count += 1
        except:
            continue
    random.shuffle(dataset)
    return dataset


def generate_data(TRAIN_SAMPLE_SIZE, TEST_SAMPLE_SIZE):
    # Train Dataset creation
    module_dir = os.path.dirname(__file__)  # Set path to current directory
    train_meta_data_file_path = os.path.join(module_dir, 'Dataset/Metadata/train-meta.xlsx')
    train_data = pd.read_excel(train_meta_data_file_path)
    train_images_file_path = os.path.join(module_dir, 'Dataset/Train/')
    # Test Dataset creation
    test_meta_data_file_path = os.path.join(module_dir, 'Dataset/Metadata/test_meta.xlsx')
    test_data = pd.read_excel(test_meta_data_file_path)
    test_images_file_path = os.path.join(module_dir, 'Dataset/Test/')

    # Encode images for training datasets
    train_images = DataPreprocessing.image_feature_extraction(train_data, train_images_file_path)
    augmentation_data = list(itertools.chain(train_images.copy(), train_images.copy(), train_images.copy()))
    augmented_images = DataPreprocessing.augment_data(augmentation_data)
    train_images_final = list(itertools.chain(train_images, augmented_images))
    np.save('DataStorage/train_data.npy', train_images_final, allow_pickle=True)
    print("TRAIN AND AUGMENTATION DATA PREPROCESSING COMPLETED..")
    # Encode images for test datasets
    test_images = DataPreprocessing.image_feature_extraction(test_data, test_images_file_path)
    np.save('DataStorage/test_data.npy', test_images, allow_pickle=True)
    print("TEST DATA PREPROCESSING COMPLETED..")
