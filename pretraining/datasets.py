import cv2
import numpy as np
import sklearn
import torch
import torchvision
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
from torchvision.transforms import v2
from torchvision.transforms.v2 import (AutoAugmentPolicy, InterpolationMode,
                                       Transform)
from torchvision.transforms.v2 import functional as F

from engine import val_step
from models import get_model

torchvision.disable_beta_transforms_warning()
import glob
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from albumentations.pytorch import ToTensorV2
from PIL import Image
from timm.data.transforms import RandomResizedCropAndInterpolation
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
from torchxrayvision.datasets import normalize


class Transforms:

    class CIFAR10:

        class ImageNet:
            train_transform = v2.Compose([
                v2.RandomResizedCrop((224, 224), scale=(0.9, 1.0), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            val_transform = v2.Compose([
                v2.Resize((224, 224), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        class Random:
            train_transform = v2.Compose([
                v2.RandomResizedCrop((224, 224), scale=(0.9, 1.0), antialias=True),
            ])    
            val_transform = v2.Compose([
                v2.Resize((224, 224), antialias=True),
                ])
    class DR:
        class ImageNet:
            train_transform = v2.Compose([
                v2.RandomResizedCrop((224, 224), scale=(0.9, 1.0), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            val_transform = v2.Compose([
                v2.Resize((224, 224), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        class Random:
            resize_transform = A.Resize(height=224, width=224, interpolation=2)  # 2 corresponds to Bilinear interpolation
            
            train_transform = A.Compose([
                resize_transform,
                A.Blur(blur_limit=3, always_apply=True),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, always_apply=True),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=180, p=0.5),
                A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.5),
                A.Transpose(p=0.5),
                A.GridDistortion(p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
                A.CLAHE(p=0.5),
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
            ])
            
            val_transform = A.Compose([
                resize_transform,
            ])
            
            tensor_transform = v2.Compose([
                v2.ToTensor(),
                v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ])

    class APTOS:
        class ImageNet:
            train_transform = v2.Compose([
                v2.RandomResizedCrop((224, 224), scale=(0.9, 1.0), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            val_transform = v2.Compose([
                v2.Resize((224, 224), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        class Random:
            resize_transform = A.Resize(height=224, width=224, interpolation=2)  # 2 corresponds to Bilinear interpolation
            
            train_transform = A.Compose([
                resize_transform,
                A.Blur(blur_limit=3, always_apply=True),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, always_apply=True),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=180, p=0.5),
                A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.5),
                A.Transpose(p=0.5),
                A.GridDistortion(p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
                A.CLAHE(p=0.5),
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
            ])
            
            val_transform = A.Compose([
                resize_transform,
            ])
            
            tensor_transform = v2.Compose([
                v2.ToTensor(),
                v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ])

        
            # combined_train_transform = v2.Compose([lambda img: train_transform(image=np.array(img))['image'], tensor_transform])
            # combined_val_transform = v2.Compose([lambda img: val_transform(image=np.array(img))['image'], tensor_transform])

    class ISIC2020:
        class ImageNet:
            train_transform = v2.Compose([
                v2.RandomResizedCrop((224, 224), scale=(0.9, 1.0), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            val_transform = v2.Compose([
                v2.Resize((224, 224), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        class Random:
            train_transform = A.Compose([
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightness(limit=0.2, p=0.75),
                A.RandomContrast(limit=0.2, p=0.75),
                A.OneOf([
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                    A.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.7),

                A.OneOf([
                    A.OpticalDistortion(distort_limit=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=1.),
                    A.ElasticTransform(alpha=3),
                ], p=0.7),

                A.CLAHE(clip_limit=4.0, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                A.Resize(224, 224, interpolation=2),
                A.Cutout(max_h_size=int(224 * 0.375), max_w_size=int(224 * 0.375), num_holes=1, p=0.7),
            ])





            val_transform = A.Compose([
                A.Resize(224, 224),
            ])

            tensor_transform = v2.Compose([
                v2.ToTensor(),
                v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ])





def get_dataset(DATASET, paths, PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK):
    
    paths = paths.split('""')
    print(paths)
    transforms = getattr(getattr(Transforms, DATASET), PRETRAINING)
    print(transforms)
    if DATASET == "Cifar10":
            ##Cifar Dataset
            trainset = torchvision.datasets.CIFAR10(root=paths[0], train=True, transform=transforms.train_transform)
            valset = torchvision.datasets.CIFAR10(root=paths[0], train=True, transform=transforms.val_transform)
            test_dataset = torchvision.datasets.CIFAR10(root=paths[0], train=False, transform=transforms.val_transform)

            idxs = np.load(paths[1]).astype('int')
            val_indices = []
            train_indices = []
            
            for i in range(len(idxs)):
                if idxs[i]:
                    val_indices.append(i)
                else:
                    train_indices.append(i)

            val_dataset = Subset(valset, val_indices)
            train_dataset = Subset(trainset, train_indices)
    elif DATASET == "MIMIC":
        ##RSNA Dataset
        train_dataset = RSNADataset(csv_file=paths[1], data_folder=paths[0], split="train", pretraining = PRETRAINING, transform=transforms.train_transform)
        val_dataset = RSNADataset(csv_file=paths[1], data_folder=paths[0], split="val", pretraining = PRETRAINING, transform=transforms.val_transform)
        test_dataset = RSNADataset(csv_file=paths[1], data_folder=paths[0], split="test", pretraining = PRETRAINING, transform=transforms.val_transform)

    elif DATASET == "ISIC":
        ##HAM Dataset
        train_dataset = HAM10000Dataset(csv_file=paths[3], data_folder=paths[0], pretraining = PRETRAINING, transform=transforms.train_transform)
        val_dataset = HAM10000Dataset(csv_file=paths[4], data_folder=paths[1], pretraining = PRETRAINING, transform=transforms.val_transform)
        test_dataset = HAM10000Dataset(csv_file=paths[5], data_folder=paths[2], pretraining = PRETRAINING, transform=transforms.val_transform)
    elif DATASET == "DR":
        ##APTOS Dataset
        train_dataset = DRDataset(csv_file=paths[1], data_folder=paths[0], split = 'train', pretraining = PRETRAINING, task = TASK, transform=transforms.train_transform, post_transform = transforms.tensor_transform)
        val_dataset = DRDataset(csv_file=paths[1], data_folder=paths[0], split = 'val', pretraining = PRETRAINING, task = TASK, transform=transforms.val_transform, post_transform = transforms.tensor_transform)
        test_dataset = DRDataset(csv_file=paths[1], data_folder=paths[0], split = 'test', pretraining = PRETRAINING, task = TASK, transform=transforms.val_transform, post_transform = transforms.tensor_transform)
    elif DATASET == "APTOS":
        ##APTOS Dataset
        train_dataset = AptosDataset(csv_file=paths[1], data_folder=paths[0], split = 'train', pretraining = PRETRAINING, task = TASK, transform=transforms.train_transform, post_transform = transforms.tensor_transform)
        val_dataset = AptosDataset(csv_file=paths[1], data_folder=paths[0], split = 'val', pretraining = PRETRAINING, task = TASK, transform=transforms.val_transform, post_transform = transforms.tensor_transform)
        test_dataset = AptosDataset(csv_file=paths[1], data_folder=paths[0], split = 'test', pretraining = PRETRAINING, task = TASK, transform=transforms.val_transform, post_transform = transforms.tensor_transform)
    elif DATASET == "ISIC2020":
        train_dataset = ISIC2020Dataset(csv_file=paths[1], data_folder=paths[0], split = 'train',  transform=transforms.train_transform, post_transform = transforms.tensor_transform)
        val_dataset = ISIC2020Dataset(csv_file=paths[1], data_folder=paths[0], split = 'val', transform=transforms.val_transform, post_transform = transforms.tensor_transform)
        test_dataset = ISIC2020Dataset(csv_file=paths[1], data_folder=paths[0], split = 'val', transform=transforms.val_transform, post_transform = transforms.tensor_transform)


    print(DATASET, len(train_dataset), len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader
    





class DRDataset(Dataset):
    def __init__(self, csv_file, data_folder, split, pretraining, task, transform=None, post_transform = None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = data_folder
        self.transform = transform
        self.split = split
        self.data = self.data.loc[self.data["split"] == self.split]
        self.task = task
        self.post_transform = post_transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        # img_name = os.path.join(self.root_dir, str(self.data.iloc[idx, 1]), str(self.data.iloc[idx, 0]))
        image = np.array(Image.open(img_name))
        label = int(self.data.iloc[idx, 1])
        if self.task == 'Regression':
            label = np.expand_dims(label, -1)
        if self.transform:
            image = self.transform(image = image)['image']
            image = self.post_transform(image)
        # print(image.max(), image.min(), label)
        return image, label




class AptosDataset(Dataset):
    def __init__(self, csv_file, data_folder, split, pretraining, task, transform=None, post_transform = None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = data_folder
        self.transform = transform
        self.split = split
        self.data = self.data.loc[self.data["split"] == self.split]
        self.task = task
        self.post_transform = post_transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data.iloc[idx, 1]), str(self.data.iloc[idx, 0]))
        image = np.array(Image.open(img_name))
        label = self.data.iloc[idx, 1]
        if self.task == 'Regression':
            label = np.expand_dims(label, -1)
        if self.transform:
            image = self.transform(image = image)['image']
            image = self.post_transform(image)
        
        return image, label


class ISIC2020Dataset(Dataset):
    def __init__(self, csv_file, data_folder, split, transform=None, post_transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = data_folder
        self.split = split
        self.data = self.data.loc[self.data["split"] == self.split]
        self.transform = transform
        self.post_transform = post_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data.iloc[idx, 1]), str(self.data.iloc[idx, 0]))
        image = np.array(Image.open(img_name))
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image = image)['image']
            image = self.post_transform(image)
        return image, label
