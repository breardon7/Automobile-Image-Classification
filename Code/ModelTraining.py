import os

import numpy as np
import pandas as pd
from keras.layers import BatchNormalization, MaxPooling2D, Conv2D, AveragePooling2D, Dropout
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical
import itertools
import DataPreprocessing


def model_definition_and_training(pretrained=True, custom=True):
    # Hyper Parameters
    IMAGE_SIZE = 224
    CHANNELS = 3
    BATCH_SIZE = 64
    MONITOR_VAL = "val_accuracy"
    TRAIN_SAMPLE_SIZE = 5000
    TEST_SAMPLE_SIZE = 1000
    LR = 1e-3
    DROPOUT = 0.5
    CLASSES = 197
    LOSS = "categorical_crossentropy"
    EPOCHS = 3

    # Train Dataset creation
    module_dir = os.path.dirname(__file__)  # Set path to current directory
    train_meta_data_file_path = os.path.join(module_dir, 'Dataset/Metadata/train-meta.xlsx')
    train_data = pd.read_excel(train_meta_data_file_path).head(TRAIN_SAMPLE_SIZE)
    train_images_file_path = os.path.join(module_dir, 'Dataset/Train/')
    # Test Dataset creation
    test_meta_data_file_path = os.path.join(module_dir, 'Dataset/Metadata/test_meta.xlsx')
    test_data = pd.read_excel(test_meta_data_file_path).head(TEST_SAMPLE_SIZE)
    test_images_file_path = os.path.join(module_dir, 'Dataset/Test/')

    # Data Augmentation
    # i changed the augmentation data to this because the mean count for each class is about 45 so we can just duplicate the whole dataset

    # Encode images for training datasets
    train_images = DataPreprocessing.image_feature_extraction(train_data, train_images_file_path)
    augmentation_data = list(itertools.chain(train_images.copy(), train_images.copy(), train_images.copy()))
    augmented_images = DataPreprocessing.augment_data(augmentation_data)
    train_images_final = list(itertools.chain(train_images, augmented_images))
    x_train = np.array([i[0] for i in train_images_final]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    y_train = np.array([i[1] for i in train_images_final])

    # Encode images for test datasets
    test_images = DataPreprocessing.image_feature_extraction(test_data, test_images_file_path)
    x_test = np.array([i[0] for i in test_images]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    y_test = np.array([i[1] for i in test_images])

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    #y_train = to_categorical(y_train)
    #y_test = to_categorical(y_test)
    #y_val = to_categorical(y_val)
    print(y_test)

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
        checkpoint = ModelCheckpoint("vgg19.h5", monitor=MONITOR_VAL, verbose=1, save_best_only=True,
                                     save_weights_only=False, save_freq=1)
        model.compile(optimizer='Adam', loss=LOSS, metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor=MONITOR_VAL, patience=5, verbose=1)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val),
                  verbose=1, callbacks=[checkpoint, early_stopping])
        model.save("SavedModel/pretrained_vgg19.h5")
        model_predict(model, x_test, y_test)

    if custom:
        model = Sequential([
            Conv2D(16, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation="relu"),
            BatchNormalization(),
            AveragePooling2D((2, 2)),
            Flatten(),
            Dense(400, activation="tanh"),
            Dropout(DROPOUT),
            BatchNormalization(),
            Dense(CLASSES, activation="softmax")
        ])
        checkpoint = ModelCheckpoint("CNN.h5", monitor=MONITOR_VAL, verbose=1, save_best_only=True,
                                     save_weights_only=False, save_freq=1)
        model.compile(optimizer='Adam', loss=LOSS, metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor=MONITOR_VAL, patience=5, verbose=1)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, validation_data=(x_test, y_test),
                  verbose=1, callbacks=[checkpoint, early_stopping])
        model.save("SavedModel/custom_model.h5")
        model_predict(model, x_test, y_test)


def model_predict(model, x_test, y_test):
    y_pred = model.predict_classes(x_test)
    print('Classification Report: Model = {}'.format(model))
    print(classification_report(y_test, y_pred))
