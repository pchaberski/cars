"""Dateset class for Stanford Cars Dataset."""


from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import torch
from importlib import import_module


class StanfordCarsDataset(Dataset):
    """Class for Stanford Cars Dataset."""

    def __init__(
        self, data_path, labels_fpath,
        image_size=[227, 227],
        convert_to_grayscale=False,
        normalize=False,
        normalization_params={'mean': None, 'std': None},
        augment_images=False,
        image_augmentations=None,
        augment_tensors=False,
        tensor_augmentations=None
    ):
        super().__init__()
        self.data_path = data_path
        self.labels_fpath = labels_fpath
        self.image_size = image_size
        self.convert_to_grayscale = convert_to_grayscale
        self.normalize = normalize
        self.normalization_params = normalization_params
        self.augment_images = augment_images
        self.image_augmentations = image_augmentations
        self.augment_tensors = augment_tensors
        self.tensor_augmentations = tensor_augmentations
        self.image_fnames = self._get_image_fnames(data_path)
        self.labels = self._get_labels(labels_fpath)

        assert len(self.image_fnames) == self.labels.shape[0], \
            f'Number of images and labels do not match: ' \
            f'{len(self.image_fnames)} != {self.labels.shape[0]}'
        assert set(self.image_fnames) == set(self.labels['image_fname'].tolist()), \
            'Image filenames do not match the list in labels data frame.'
        if self.convert_to_grayscale:
            assert len(normalization_params['mean']) == len(normalization_params['std']) == 1, \
                f'`mean` and `std` normalization params has to be of length 1 for grayscale'
        else:
            assert len(normalization_params['mean']) == len(normalization_params['std']) == 3, \
                f'`mean` and `std` normalization params has to be of length 3 for RGB'

    def _transform(self, image):
        # Resizing
        transf_list = [transforms.Resize(self.image_size)]

        # Optional grayscale conversion
        if self.convert_to_grayscale:
            transf_list += [transforms.Grayscale()]

        # Optional augmentations on image - training only
        if self.augment_images:
            transf_list += self._parse_and_load_transforms(self.image_augmentations)

        # Convert to tensor
        transf_list += [transforms.ToTensor()]

        # Optional augmentations on tensor - training only
        if self.augment_tensors:
            transf_list += self._parse_and_load_transforms(self.tensor_augmentations)

        # Optional normalization
        if self.normalize:
            transf_list += [transforms.Normalize(
                mean=self.normalization_params['mean'],
                std=self.normalization_params['std']
            )]

        transf = transforms.Compose(transf_list)

        return transf(image)

    def _get_image_fnames(self, data_path):
        jpg_fnames = [file for file in os.listdir(data_path) if file.endswith('.jpg')]

        return jpg_fnames

    def _get_labels(self, labels_fpath):
        df_labels = pd.read_csv(labels_fpath)[['image_fname', 'class']]

        return df_labels

    def _parse_and_load_transforms(self, tranforms_dict):
        _transf_list = []
        for k, v in tranforms_dict.items():
            _transf = getattr(import_module('torchvision.transforms'), k)
            _transf_list += [_transf(**v)]

        return _transf_list

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_fpath = os.path.join(self.data_path, image_fname)
        image = Image.open(image_fpath).convert('RGB')
        image = self._transform(image)
        """This returns class indexes from range [0, C-1]
        as accepted during loss calculation, real class ids are from [1, C]"""
        label = torch.as_tensor(
            self.labels[self.labels['image_fname'] == image_fname]['class'].values[0] - 1,
            dtype=torch.long)

        return image, label
