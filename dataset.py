import glob
import os
import cv2

from torch.utils.data import Dataset

import numpy as np
from torchxrayvision.datasets import normalize
import torch

import pandas as pd
from PIL import Image

class DogsCats(Dataset):
    """Dogs vs Cats dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform

        self.images = []
        self.labels = []

        for i in range(2):
            for image_path in glob.glob(os.path.join(self.root_dir, str(i), "*")):
                self.images.append(image_path)
                self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]

        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = image / 255.0
        
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float)
        if self.transform:
            image = self.transform(image)

        return image, label

class DogsCatsTest(Dataset):
    """Dogs vs Cats dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform

        self.images = []

        for image_path in glob.glob(os.path.join(self.root_dir, "*")):
            self.images.append(image_path)
      

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]

        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = image / 255.0

        if self.transform:
            image = self.transform(image)

        return image, image_name.split("/")[-1].split(".")[0]


class RSNADataset(Dataset):
    def __init__(self, csv_file, data_folder, split, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data_folder = data_folder
        self.transform = transform
        self.split = split
        # self.data = self.data[self.data['class'] != 'No Lung Opacity / Not Normal']

        # Filter the data based on the specified split
        self.data = self.data[self.data['split'] == split]
        # print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print('trainnnnnnnnnnnnnn',self.data.iloc[idx, 8])
        image_folder = os.path.join(self.data_folder, 'train_v3')
        # print('imageeeeeeeeeee', image_folder)
        image_path = os.path.join(image_folder, self.data.iloc[idx, 1] + '.pt')
        # print(image_path)
        image = torch.load(image_path)#.convert('L')
        target = int(self.data.iloc[idx, 7])
        image = normalize(image, maxval=255, reshape=True)

        # print(target)
        if self.transform:
            image = self.transform(image)
        return image, target