import os

import numpy as np
import pandas as pd
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.models import Sequential
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils.np_utils import to_categorical

from Code import DataPreprocessing

IMAGE_SIZE = 224
CHANNELS = 3
BATCH_SIZE = 64
MONITOR_VAL = "val_accuracy"
SAMPLE_SIZE = 1000

module_dir = os.path.dirname(__file__)
train_meta_data_file_path = os.path.join(module_dir, 'Dataset/Metadata/train-meta.xlsx')
train_data = pd.read_excel(train_meta_data_file_path).head(SAMPLE_SIZE)
train_images_file_path = os.path.join(module_dir, 'Dataset/Train/')

test_meta_data_file_path = os.path.join(module_dir, 'Dataset/Metadata/test_meta.xlsx')
test_data = pd.read_excel(test_meta_data_file_path).head(SAMPLE_SIZE)
test_images_file_path = os.path.join(module_dir, 'Dataset/Test/')

train_images = DataPreprocessing.image_feature_extraction(train_data, train_images_file_path, IMAGE_SIZE)
x_train = np.array([i[0] for i in train_images]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
y_train = np.array([i[1] for i in train_images])
print(len(y_train))
test_images = DataPreprocessing.image_feature_extraction(test_data, test_images_file_path, IMAGE_SIZE)
x_test = np.array([i[0] for i in test_images]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
y_test = np.array([i[1] for i in test_images])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# test run to make sure xtest and y test are valid
loss = keras.losses.CategoricalCrossentropy(
    from_logits=False, label_smoothing=0.0, axis=-1,
    reduction=losses_utils.ReductionV2.AUTO,
    name='categorical_crossentropy'
)
vgg = VGG19(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
            classes=y_train.shape[1])
for layer in vgg.layers:
    layer.trainable = False
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(197, activation="softmax"))
model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
checkpoint = ModelCheckpoint("vgg19.h5", monitor=MONITOR_VAL, verbose=1, save_best_only=True,
                             save_weights_only=False, period=1)
early_stopping = EarlyStopping(monitor=MONITOR_VAL, patience=5, verbose=1)
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, validation_data=(x_test, y_test),
          verbose=1, callbacks=[checkpoint, early_stopping])

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))