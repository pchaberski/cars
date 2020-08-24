"""Dateset class for Stanford Cars Dataset."""


from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import torch


class StanfordCarsDataset(Dataset):
    """Class for Stanford Cars Dataset."""

    def __init__(self, data_path, labels_fpath, image_size=(227, 227)):
        super().__init__()
        self.data_path = data_path
        self.labels_fpath = labels_fpath
        self.image_fnames = self._get_image_fnames(data_path)
        self.image_size = image_size
        self.labels = self._get_labels(labels_fpath)

        assert len(self.image_fnames) == self.labels.shape[0], \
            f'Number of images and labels do not match: ' \
            f'{len(self.image_fnames)} != {self.labels.shape[0]}'
        assert set(self.image_fnames) == set(self.labels['image_fname'].tolist()), \
            'Image filenames do not match the list in labels data frame.'

    def _transform(self, image):
        transf = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        return transf(image)

    def _get_image_fnames(self, data_path):
        jpg_fnames = [file for file in os.listdir(data_path) if file.endswith('.jpg')]

        return jpg_fnames

    def _get_labels(self, labels_fpath):
        df_labels = pd.read_csv(labels_fpath)[['image_fname', 'class']]

        return df_labels

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
            self.labels[self.labels['image_fname'] == image_fname]['class'].values[0] - 1)

        return image, label
