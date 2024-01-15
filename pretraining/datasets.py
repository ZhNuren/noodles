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




class MIMIC_Dataset(Dataset):
    """MIMIC-CXR Dataset
 
    Citation:
 
    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY,
    Mark RG, Horng S. MIMIC-CXR: A large publicly available database of
    labeled chest radiographs. arXiv preprint arXiv:1901.07042. 2019 Jan 21.
 
    https://arxiv.org/abs/1901.07042
 
    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """
 
    def __init__(self,
                 imgpath,
                 csvpath,
                 metacsvpath,
                 splitpath,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 seed=0,
                 split = 'train',
                 unique_patients=True
                 ):
 
        super(MIMIC_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
 
        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            "Support Devices"]
 
        self.pathologies = sorted(self.pathologies)
 
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = metacsvpath
        self.splitpath = splitpath
        # self.PIL_transform = transforms.ToPILImage()
       
        self.metacsv = pd.read_csv(self.metacsvpath)
        # print('metaaaaaaaaaaaaaaa',self.metacsv)
        self.split_dataset = pd.read_csv(self.splitpath)
        test_df = self.split_dataset[(self.split_dataset['split'] == split)]
        test_df.reset_index(drop=True, inplace=True)
        # print('testttttttttttttttt',test_df)
 
        final_df = pd.merge(test_df, self.metacsv, on=['dicom_id', 'subject_id', 'study_id'], how='inner')
        final_df = final_df[self.metacsv.columns]
 
        self.csv = self.csv.set_index(['subject_id', 'study_id'])
        final_df = final_df.set_index(['subject_id', 'study_id'])
        print('hiiiiiiiiiiiiiiiiiii')
        self.csv = self.csv.join(final_df, how='inner').reset_index()
        # Keep only the desired view
        self.csv["view"] = self.csv["ViewPosition"]
        self.limit_to_selected_views(views)
 
        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()
        print('helllllllllllooooooooooooo')
 
        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
 
            labels.append(mask.values)
        print('byereeeeeeeeeeee')
 
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)
 
        self.labels[self.labels == -1] = np.nan
        # print(self.labels.shape)
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))
        print('edfsdgfhgfrhtdfghgfd')
 
 
        # offset_day_int
        self.csv["offset_day_int"] = self.csv["StudyDate"]
 
        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str)
        # print('final df', self.csv)
 
    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
 
        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])
 
        img_path = os.path.join(self.imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")
        # img_path = os.path.join(self.imgpath, dicom_id + '.jpg' + '_' + 'p' + subjectid[:2] + '_' + 'p' + subjectid + '_' + 's' + studyid + '_' + 'GT_img1' + '.jpeg')
        # print(img_path)
        img = imread(img_path)
        # img = Image.fromarray(img)
        img = np.expand_dims(img,axis=0)
        # img = imread(img_path)
        img = torch.from_numpy(img)
        img = torch.cat([img, img, img], dim=0)
        sample["img"] = img
        print('beforeeeee', sample["img"].shape)
        # img = img.detach().cpu().numpy()
        # print(img.shape)
        # sample["img"] = normalize(img, maxval=255, reshape=False)
        # print(sample["img"], sample["img"].shape)
        # sample = apply_transforms(sample, self.transform)
        # sample['img'] = sample['img'].transpose(1,2,0)
        sample = apply_transforms(sample, self.data_aug)
        print(sample["img"].shape)
        # print(sample["img"], sample["img"].shape)
        return sample