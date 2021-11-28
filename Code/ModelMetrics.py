import os

import numpy as np
import pandas as pd
from keras.layers import BatchNormalization, MaxPooling2D, Conv2D, AveragePooling2D, Dropout
from sklearn.metrics import classification_report
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.resnet import ResNet101
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef


import DataPreprocessing

# Hyper Parameters
IMAGE_SIZE = 224
CHANNELS = 3
BATCH_SIZE = 64
MONITOR_VAL = "val_accuracy"
SAMPLE_SIZE = 8000
LR = 1e-3
DROPOUT = 0.5
CLASSES = 197
LOSS = "categorical_crossentropy"

# Train Dataset creation
module_dir = os.path.dirname(__file__)  # Set path to current directory
train_meta_data_file_path = os.path.join(module_dir, 'Dataset/Metadata/train-meta.xlsx')
train_data = pd.read_excel(train_meta_data_file_path).head(SAMPLE_SIZE)
train_images_file_path = os.path.join(module_dir, 'Dataset/Train/')

# Test Dataset creation
test_meta_data_file_path = os.path.join(module_dir, 'Dataset/Metadata/test_meta.xlsx')
test_data = pd.read_excel(test_meta_data_file_path).head(SAMPLE_SIZE)
test_images_file_path = os.path.join(module_dir, 'Dataset/Test/')

# Encode images for training datasets
train_images = DataPreprocessing.image_feature_extraction(train_data, train_images_file_path, IMAGE_SIZE, False)
# train_images_aug = DataPreprocessing.image_augmentation(train_data_aug, train_images_file_path, IMAGE_SIZE)
x_train = np.array([i[0] for i in train_images]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
y_train = np.array([i[1] for i in train_images])

# Encode images for test datasets
test_images = DataPreprocessing.image_feature_extraction(test_data, test_images_file_path, IMAGE_SIZE, False)
x_test = np.array([i[0] for i in test_images]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
y_test = np.array([i[1] for i in test_images])

# Data Augmentation
# Add value counts of targets as column
target_counts = train_data['class'].value_counts()
target_counts_dict = target_counts.to_dict()
train_data_copy = train_data.copy()
train_data_copy['Counts'] = train_data_copy['class'].map(target_counts_dict)
print(train_data_copy)

# Create train dataset for augmentation
train_data_1 = train_data_copy[train_data_copy['Counts'] == 1].copy()
train_data_1 = pd.concat([train_data_1]*10, ignore_index=True)
train_data_2 = train_data_copy[train_data_copy['Counts'] == 2].copy()
train_data_2 = pd.concat([train_data_2]*5, ignore_index=True)
train_data_3 = train_data_copy[train_data_copy['Counts'] == 3].copy()
train_data_3 = pd.concat([train_data_3]*3, ignore_index=True)
train_data_4 = train_data_copy[train_data_copy['Counts'] >= 4].copy()
train_data_4 = pd.concat([train_data_4]*2, ignore_index=True)
frames = [train_data_1,train_data_2,train_data_3,train_data_4]
train_data_aug = pd.concat(frames)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def model_definition(model_type, opt, lr, batch):
    if opt == 'Adam':
        optimzer = keras.optimizers.Adam(learning_rate=lr)
    elif opt == 'RMSprop':
        optimzer = keras.optimizers.RMSprop(learning_rate=lr)
    elif opt == 'SGD':
        optimzer = keras.optimizers.SGD(learning_rate=lr)
    elif opt == 'Adadelta':
        optimzer = keras.optimizers.Adadelta(learning_rate=lr)

    if model_type == 'cnn':
        # // CNN model
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
        mod_name = model_type  + '_' + opt  + '_' + 'batch' + batch + 'lr' + str(lr).replace('0.', '').replace('.','') + '.h5'
        checkpoint = ModelCheckpoint(mod_name, monitor=MONITOR_VAL, verbose=1, save_best_only=True,
                                     save_weights_only=False, save_freq=1)

        model.compile(optimizer=optimizer, loss=LOSS, metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor=MONITOR_VAL, patience=5, verbose=1)
        model.fit(x_train, y_train, batch_size=batch, epochs=20, validation_data=(x_test, y_test),
                  verbose=1, callbacks=[checkpoint, early_stopping])

    else:
        if model_type == 'vgg':
            mod_specify = VGG19(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
                    classes=CLASSES)
        elif model_type == 'resnet':
            mod_specify = ResNet101(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
                                classes=CLASSES)
        elif model_type == 'densenet':
            mod_specify = DenseNet201(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
                                classes=CLASSES)
        elif model_type == 'inception':
            mod_specify = InceptionV3(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
                                classes=CLASSES)
        mod_name = model_type + '_' + opt  + '_' + 'batch' + batch + 'lr' + str(lr).replace('0.', '').replace('.', '') + '.h5'
        for layer in vgg.layers:
            layer.trainable = False
        model = Sequential()
        model.add(mod_specify)
        model.add(Flatten())
        model.add(Dense(CLASSES, activation="softmax"))
        checkpoint = ModelCheckpoint(mod_name, monitor=MONITOR_VAL, verbose=1, save_best_only=True,
                                     save_weights_only=False, save_freq=1)
        model.compile(optimizer=optimzer, loss=LOSS, metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor=MONITOR_VAL, patience=5, verbose=1)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=20, validation_data=(x_test, y_test),
                  verbose=1, callbacks=[checkpoint, early_stopping])


def model_predict(model):
    y_pred = model.predict(x_test)

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    acc = accuracy_metric(y_test, y_pred)
    hlm = hamming_metric(y_test, y_pred)
    avg = (acc + hlm) / 2

    return acc, hlm, avg


def run_models(model_list, learning_rate_list, batch_list, optimizer_list):
    df = pd.DataFrame(columns=['model', 'optimizer', 'learning_rate', 'batch_size', 'accuracy', 'hlm', 'metrics_avg'])

    for m in model_list:
        for opt in optimizer_list:
            for lr in learning_rate_list:
                for batch in batch_list:
                    mod = model_definition(m, opt, lr, batch)
                    acc, hlm, avg = model_predict(mod)
                    df.loc[len(df)] = [m, opt, lr, batch, acc, hlm, avg]

    df.to_csv('model_metrics.csv')
    
model_list = ['cnn', 'vgg', 'resnet', 'densenet', 'inception']
optimizer_list = ['Adam', 'RMSprop', 'SGD', 'Adadelta']
learning_rate_list = [0.00001, 0.00005, 0.0001, 0.001]
batch_list = [32, 64]

run_models(model_list=model_list, learning_rate_list=learning_rate_list, batch_list=batch_list, optimzer_list=optimizer_list)