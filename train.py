from engine import trainer, val_step
from utils import plot_results
from models import get_model

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import argparse

from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform

from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data.dataset import Subset

from utils import EarlyStopper

import yaml
import json
import time
import os
import wandb
from dataset import RSNADataset, HAM10000Dataset,AptosDataset
from utils import load_model


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/config_init.yaml', metavar='DIR', help='configs')

args = parser.parse_args()


config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
print(config)
run = wandb.init(entity='biomed', project='model_soups', config=config)


LEARNING_RATE = float(config["LEARNING_RATE"])
LEARNING_SCHEDULER = config["LEARNING_SCHEDULER"]
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])
NUM_CLASSES = int(config["NUM_CLASSES"])
LINEAR_PROBING = config["LINEAR_PROBING"]
PROBING_EPOCHS = int(config["PROBING_EPOCHS"])
PATIENCE = int(config["PATIENCE"])

LOSS = config["LOSS"]

IMAGE_SIZE = int(config["IMAGE_SIZE"])
MODEL = config["MODEL"]
PRETRAINED = config["PRETRAINED"]



DATASET = config["DATASET"]
RSNA_CSV = config["RSNA_CSV"]
RSNA_PATH = config["RSNA_PATH"]
CIFAR_PATH = config["CIFAR_PATH"]
CIFAR_INDICES = config["CIFAR_INDICES"]

HAM_TRAIN_CSV = str(config["HAM_TRAIN_CSV"])
HAM_VAL_CSV = str(config["HAM_VAL_CSV"])
HAM_TEST_CSV = str(config["HAM_TEST_CSV"])
HAM_TRAIN_FOLDER = str(config["HAM_TRAIN_FOLDER"])
HAM_VAL_FOLDER = str(config["HAM_VAL_FOLDER"])
HAM_TEST_FOLDER = str(config["HAM_TEST_FOLDER"])


APTOS_CSV = str(config["APTOS_CSV"])
APTOS_FOLDER = str(config["APTOS_FOLDER"])



PRETRAINING = config["PRETRAINING"]
SAVE_DIR = config["SAVE_DIR"]

CUDA_DEVICE = int(config["CUDA_DEVICE"])

DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else 'cpu')

print(f"Using {DEVICE} device")

def START_seed(seed_value=9):
    seed = seed_value
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 


def main():
    START_seed()

    #run id is date and time of the run
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    #create folder for this run in runs folder
    os.mkdir(SAVE_DIR + run_id)

    save_dir = SAVE_DIR + run_id
    

    #load data
    if PRETRAINING == "ImageNet":
        train_transform = v2.Compose([
            v2.RandomRotation(degrees=(-70, 70)),
            v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.2), antialias=True, interpolation = InterpolationMode.BILINEAR),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ])

        val_transform = transforms.Compose([
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            ])
    else:
        train_transform = v2.Compose([
            v2.RandomRotation(degrees=(-70, 70)),
            v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.2), antialias=True, interpolation = InterpolationMode.BILINEAR),
        ])

        val_transform = transforms.Compose([
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            ])


    if DATASET == "Cifar10":
        ##Cifar Dataset
        trainset = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=True, transform=train_transform, download=True)
        valset = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=True, transform=val_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=False, transform=val_transform, download=True)

        # train_size = int(0.9 * len(trainset))
        # val_size = len(trainset) - train_size
        
        # generator1 = torch.Generator().manual_seed(9)
        # generator2 = torch.Generator().manual_seed(9)

        # _, val_dataset = torch.utils.data.random_split(valset, [train_size, val_size], generator = generator1)
        # train_dataset, _ = torch.utils.data.random_split(trainset, [train_size, val_size], generator = generator2)

        idxs = np.load(CIFAR_INDICES).astype('int')
        val_indices = []
        train_indices = []
        
        for i in range(len(idxs)):
            if idxs[i]:
                val_indices.append(i)
            else:
                train_indices.append(i)

        val_dataset = Subset(valset, val_indices)
        train_dataset = Subset(trainset, train_indices)
        
    
    elif DATASET == "Rsna":
        ##RSNA Dataset
        train_dataset = RSNADataset(csv_file=RSNA_CSV, data_folder=RSNA_PATH, split="train", pretraining = PRETRAINING, transform=train_transform)
        val_dataset = RSNADataset(csv_file=RSNA_CSV, data_folder=RSNA_PATH, split="val", pretraining = PRETRAINING, transform=val_transform)
        test_dataset = RSNADataset(csv_file=RSNA_CSV, data_folder=RSNA_PATH, split="test", pretraining = PRETRAINING, transform=val_transform)

    elif DATASET == "HAM":
        ##RSNA Dataset
        train_dataset = HAM10000Dataset(csv_file=HAM_TRAIN_CSV, data_folder=HAM_TRAIN_FOLDER, pretraining = PRETRAINING, transform=train_transform)
        val_dataset = HAM10000Dataset(csv_file=HAM_VAL_CSV, data_folder=HAM_VAL_FOLDER, pretraining = PRETRAINING, transform=val_transform)
        test_dataset = HAM10000Dataset(csv_file=HAM_TEST_CSV, data_folder=HAM_TEST_FOLDER, pretraining = PRETRAINING, transform=val_transform)

    elif DATASET == "APTOS":
        ##RSNA Dataset
        train_dataset = AptosDataset(csv_file=APTOS_CSV, data_folder=APTOS_FOLDER, split = 'train', pretraining = PRETRAINING, transform=train_transform)
        val_dataset = AptosDataset(csv_file=APTOS_CSV, data_folder=APTOS_FOLDER, split = 'val', pretraining = PRETRAINING, transform=val_transform)
        test_dataset = AptosDataset(csv_file=APTOS_CSV, data_folder=APTOS_FOLDER, split = 'test', pretraining = PRETRAINING, transform=val_transform)

    print(len(val_dataset), len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
    #load model
    model = get_model(MODEL, PRETRAINED, num_classes=NUM_CLASSES)

    model.to(DEVICE)
    torch.compile(model)
    
    #load optimizer
    if LOSS == "MSE":
        loss = torch.nn.MSELoss()
    elif LOSS == "L1Loss":
        loss = torch.nn.L1Loss()
    elif LOSS == "SmoothL1Loss":
        loss = torch.nn.SmoothL1Loss()
    elif LOSS == "CrossEntropyLoss":
        loss = torch.nn.CrossEntropyLoss()
    elif LOSS == "BCEWithLogitsLoss":
        loss = torch.nn.BCEWithLogitsLoss()
        
    else:
        raise Exception("Loss not implemented")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if LEARNING_SCHEDULER == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, verbose=True)
    elif LEARNING_SCHEDULER == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    elif LEARNING_SCHEDULER == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
    elif LEARNING_SCHEDULER == "MultiStepLR":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20], gamma=0.1)
    else:
        lr_scheduler = None

    early_stopper = EarlyStopper(patience=PATIENCE, min_delta=0.001)

    if LINEAR_PROBING:
        linear_probing_epochs = PROBING_EPOCHS
    else:
        linear_probing_epochs = None
     
    #train model
    results = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_name=LEARNING_SCHEDULER,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        save_dir=save_dir,
        early_stopper=early_stopper,
        linear_probing_epochs=linear_probing_epochs
    )

    
    checkpoint = torch.load(save_dir + "/best_checkpoint.pth")
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)
    torch.compile(model)

    test_loss, test_acc, test_f1, test_recall = val_step(model, test_loader, loss, DEVICE)

    config["test_acc"] = test_acc
    config["test_loss"] = test_loss
    config["test_f1"] = test_f1
    config["test_recall"] = test_recall
    

    wandb.log({"test_loss": test_loss, "test_acc": test_acc})

    train_summary = {
        "config": config,
        "results": results,
    }


    with open(save_dir + "/train_summary.json", "w") as f:
        json.dump(train_summary, f, indent=4)

    plot_results(results, save_dir)

    



if __name__ == "__main__":
    main()


