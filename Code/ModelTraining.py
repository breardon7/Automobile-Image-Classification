import warnings
from sys import platform

import numpy as np
from keras.layers import BatchNormalization, MaxPooling2D, Conv2D, AveragePooling2D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.utils.np_utils import to_categorical

from Code.ModelMetrics import model_predict

warnings.filterwarnings('ignore')
# i added this security line for mac users because the keychain access blocks unverified ssl from the dataset url
if platform == "darwin":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context


def model_definition_and_training(pretrained, custom):
    # Hyper Parameters
    IMAGE_SIZE = 224
    CHANNELS = 3
    BATCH_SIZE = 64
    MONITOR_VAL = "val_accuracy"
    DROPOUT = 0.5
    LOSS = "categorical_crossentropy"
    EPOCHS = 3
    CLASSES = 197

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

    if custom:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(AveragePooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(400, activation="relu"))
        model.add(Dropout(DROPOUT))
        model.add(BatchNormalization())
        model.add(Dense(CLASSES, activation="softmax"))
        checkpoint = ModelCheckpoint("SavedModel/custom_model.h5", monitor=MONITOR_VAL, verbose=1, save_best_only=True,
                                     save_weights_only=False, save_freq=1)
        model.compile(optimizer='Adam', loss=LOSS, metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor=MONITOR_VAL, patience=5, verbose=1)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, validation_data=(x_val, y_val),
                  verbose=1, callbacks=[checkpoint, early_stopping])
        model_predict(model)
