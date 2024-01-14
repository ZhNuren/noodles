import glob
import os
import cv2

from torch.utils.data import Dataset
import torchxrayvision as xrv
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
    def __init__(self, csv_file, data_folder, split, pretraining, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data_folder = data_folder
        self.transform = transform
        self.split = split
        # self.data = self.data[self.data['class'] != 'No Lung Opacity / Not Normal']
        self.pretraining = pretraining
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
        # image = np.array(torch.load(image_path))
        image = torch.load(image_path)#.convert('L')
        target = int(self.data.iloc[idx, 7])
        # image = normalize(image, maxval=255, reshape=False)
        # print(target)
        # image = xrv.datasets.ToPILImage()(image)
    
        if self.transform:
            image = self.transform(image)

        if self.pretraining != "ImageNet":
            image = normalize(np.array(image), maxval=255, reshape=True)        
        # print(np.array(image))
        return image, target
    


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, data_folder, pretraining, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = data_folder
        self.transform = transform
        self.pretraining = pretraining

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx, 0]
        img_path = f"{self.root_dir}/{img_name}.jpg"
        image = Image.open(img_path)

        label = self.data_frame.iloc[idx, 1:].values.astype(np.float32)
        label = np.argmax(label)

        if self.transform:
            image = self.transform(image)
        return image, label




class AptosDataset(Dataset):
    def __init__(self, csv_file, data_folder, split, pretraining, task, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = data_folder
        self.transform = transform
        self.split = split
        self.data = self.data.loc[self.data["split"] == self.split]
        self.task = task
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data.iloc[idx, 1]), str(self.data.iloc[idx, 0]))
        image = Image.open(img_name)
        label = self.data.iloc[idx, 1]
        if self.task == 'Regression':
            label = np.expand_dims(label, -1)
        if self.transform:
            image = self.transform(image)
        
        return image, label
