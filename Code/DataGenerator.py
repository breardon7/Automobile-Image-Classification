import itertools
import os

import numpy as np
import pandas as pd

from Code import DataPreprocessing


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


generate_data(1000, 1000)
