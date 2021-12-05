import numpy as np
from keras.layers import BatchNormalization, MaxPooling2D, Conv2D, AveragePooling2D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical


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
        checkpoint = ModelCheckpoint("vgg19.h5", monitor=MONITOR_VAL, verbose=1, save_best_only=True,
                                     save_weights_only=False, save_freq=1)
        model.compile(optimizer='Adam', loss=LOSS, metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor=MONITOR_VAL, patience=5, verbose=1)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val),
                  verbose=1, callbacks=[checkpoint, early_stopping])
        model.save("SavedModel/pretrained_vgg19.h5")

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
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, validation_data=(x_val, y_val),
                  verbose=1, callbacks=[checkpoint, early_stopping])
        model.save("SavedModel/custom_model.h5")

