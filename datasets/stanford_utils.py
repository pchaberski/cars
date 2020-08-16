"""
Utility for Stanford data preparation.

Dataset: https://ai.stanford.edu/~jkrause/cars/car_dataset.html

Assuming that there are 3 components of Stanford Cars Dataset downloaded to \
a common folder which location is provided in `config.yml` \
(parameter: `stanford_raw_data_path`):
    - `car_ims.tgz` - updated collection of train and test images \
        (http://imagenet.stanford.edu/internal/car196/car_ims.tgz)
    - `cars_annos.mat` - updated train and test labels and bounding boxes \
        (http://imagenet.stanford.edu/internal/car196/cars_annos.mat)
    - `car_devkit.tgz` - original devkit containing class names \
        (https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz)
"""


import pandas as pd
import numpy as np
from mat4py import loadmat
import tarfile
import os
from tqdm import tqdm
from utils.timer import timer


@timer
def prepare_stanford_dataset(raw_data_path, dest_data_path='input/stanford', **kwargs):
    """
    Prepare data and metadata for modeling.

    :param raw_data_path: Path to the folder containing: car_ims.tgz, cars_annos.mat, car_devkit.tgz
    :type raw_data_path: str
    :param dest_data_path: Path to the destination folder, defaults to 'input/stanford'
    :type dest_data_path: str, optional
    :Keyword Arguments:
        * *get_time* (bool) --
            If True, execution time is being measured
        * *logger* (logging.Logger) --
            If a valid logger is provided, time is logged instead of printing
    """
    assert os.path.isdir(raw_data_path), f'Provided directory does not exist: {raw_data_path}'
    raw_files = ['car_ims.tgz', 'cars_annos.mat', 'car_devkit.tgz']
    for raw_file in raw_files:
        assert os.path.isfile(os.path.join(raw_data_path, raw_file)), f'File does not exist: {raw_file}'

    # Get original annotation data frame
    df_annos = pd.DataFrame(
        loadmat(os.path.join(raw_data_path, 'cars_annos.mat'))['annotations'])

    train_fpaths = list(df_annos[df_annos['test'] == 0]['relative_im_path'])
    test_fpaths = list(df_annos[df_annos['test'] == 1]['relative_im_path'])

    # Extract train and test images to separate folders
    tar_ims = tarfile.open(os.path.join(raw_data_path, 'car_ims.tgz'), "r:gz")
    for member in tqdm(tar_ims.getmembers()):
        if member.isreg() and member.name in train_fpaths:
            member.name = os.path.basename(member.name)
            tar_ims.extract(member, os.path.join(dest_data_path, 'train'))
        elif member.isreg() and member.name in test_fpaths:
            member.name = os.path.basename(member.name)
            tar_ims.extract(member, os.path.join(dest_data_path, 'test'))
    tar_ims.close()

    # Get label names
    tar_labels = tarfile.open(os.path.join(raw_data_path, 'car_devkit.tgz'), "r:gz")
    mat_labels = loadmat(tar_labels.extractfile(tar_labels.getmember('devkit/cars_meta.mat')))
    df_labels = pd.DataFrame({
        'class': np.arange(1, len(mat_labels['class_names']) + 1),
        'class_name': mat_labels['class_names']
    })

    # Create output data frames
    df_train = pd.DataFrame({
        'image_fname': df_annos[df_annos['test'] == 0]['relative_im_path'].apply(os.path.basename),
        'class': df_annos[df_annos['test'] == 0]['class'],
        'bbox_x1': df_annos[df_annos['test'] == 0]['bbox_x1'],
        'bbox_y1': df_annos[df_annos['test'] == 0]['bbox_y1'],
        'bbox_x2': df_annos[df_annos['test'] == 0]['bbox_x2'],
        'bbox_y2': df_annos[df_annos['test'] == 0]['bbox_y2']
    })
    df_test = pd.DataFrame({
        'image_fname': df_annos[df_annos['test'] == 1]['relative_im_path'].apply(os.path.basename),
        'class': df_annos[df_annos['test'] == 1]['class'],
        'bbox_x1': df_annos[df_annos['test'] == 1]['bbox_x1'],
        'bbox_y1': df_annos[df_annos['test'] == 1]['bbox_y1'],
        'bbox_x2': df_annos[df_annos['test'] == 1]['bbox_x2'],
        'bbox_y2': df_annos[df_annos['test'] == 1]['bbox_y2']
    })
    df_train = df_train.merge(df_labels, how='inner', on='class')
    df_test = df_test.merge(df_labels, how='inner', on='class')

    # Save metadata frames to CSV
    df_train.to_csv(os.path.join(dest_data_path, 'train_labels.csv'), index=False)
    df_test.to_csv(os.path.join(dest_data_path, 'test_labels.csv'), index=False)
