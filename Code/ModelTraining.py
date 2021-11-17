import os

import numpy as np
import pandas as pd

from Code import DataPreprocessing
IMAGE_SIZE = 224
CHANNELS = 3
BATCH_SIZE = 64
MONITOR_VAL = "val_accuracy"
SAMPLE_SIZE = 100

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

test_images = DataPreprocessing.image_feature_extraction(test_data, test_images_file_path, IMAGE_SIZE)
x_test = np.array([i[0] for i in test_images]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
y_test = np.array([i[1] for i in test_images])
