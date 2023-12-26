from engine import trainer, val_step
from utils import plot_results
from model import get_model

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

from dataset import RSNADataset

from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()

import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

from utils import EarlyStopper

import yaml
import json
import time
import os
import wandb
from utils import load_model


config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

run = wandb.init(entity='biomed', project='model_soups', config=config)


wandconf = {
        "LEARNING_SCHEDULER": config["LEARNING_SCHEDULER"],
        "BATCH_SIZE": config["BATCH_SIZE"],
        "IMAGE_SIZE": config["IMAGE_SIZE"],
        "MODEL": config["MODEL"],
        "PRETRAINED": config["PRETRAINED"],
        "LOSS": config["LOSS"],
}

LEARNING_RATE = float(config["LEARNING_RATE"])
LEARNING_SCHEDULER = config["LEARNING_SCHEDULER"]
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])
LINEAR_PROBING = config["LINEAR_PROBING"]
PROBING_EPOCHS = int(config["PROBING_EPOCHS"])
PATIENCE = int(config["PATIENCE"])

LR_RATE_LIST = config["LR_RATE_LIST"]
NUM_EPOCHS_LIST = config["NUM_EPOCHS_LIST"]
AUGMENT_LIST = config["AUGMENT_LIST"]

LOSS = config["LOSS"]

IMAGE_SIZE = int(config["IMAGE_SIZE"])
MODEL = config["MODEL"]
PRETRAINED = config["PRETRAINED"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")


def START_seed():
    seed = 9
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 


def main():
    START_seed()

    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

    if not os.path.exists("./runs"):
        os.mkdir("./runs")
    #create folder for this run in runs folder

    os.mkdir("./runs/finetune_" + run_id)

    parent_dir = "./runs/finetune_" + run_id

    # RUN_NAME = "2023-12-15_16-37-34"
    RUN_NAME = "2023-12-15_16-34-00"

    run_path = f"runs/{RUN_NAME}/"

    
    
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
    

    #create pandas dataframe to store results
    resultsexp = pd.DataFrame(columns=["lr_rate", "num_epochs", "augmentation", "test_acc", "test_loss"])
    

    for lr_rate in LR_RATE_LIST:
        for num_epoch in NUM_EPOCHS_LIST:
            for augment in AUGMENT_LIST:

                lr_rate = float(lr_rate)
                num_epoch = int(num_epoch)

                wandconf["LEARNING_RATE"] = "{:.2e}".format(lr_rate)
                wandconf["NUM_EPOCHS"] = num_epoch
                wandconf["AUGMENTATION"] = augment
                

                model = get_model(MODEL, PRETRAINED, num_classes=2)
                model = load_model(model, run_path + "best_model.pt")
                model.to(DEVICE)
                torch.compile(model)

                #run id is date and time of the run
                run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
                os.mkdir(parent_dir + "/hparam_" + run_id)
                save_dir = parent_dir + "/hparam_" + run_id

                if augment == "Minimal":
                    train_transform = v2.Compose([
                        v2.ToTensor(),
                        v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.9, 1.0), antialias=True),
                        v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                    ])
                elif augment == "Medium":
                    train_transform = v2.Compose([
                        v2.ToTensor(),
                        v2.RandomRotation(degrees=(-70, 70)),
                        v2.RandomAffine(degrees=(-15, 15), translate=(0.25, 0.25), scale=(0.7, 1.2), shear=(-15, 15, -15, 15)),
                        v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.9, 1.0), antialias=True),
                        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                        v2.RandomEqualize(p=0.2),
                        v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                    ])
                elif augment == "Heavy":
                    train_transform = v2.Compose([
                        v2.ToTensor(),
                        v2.RandomRotation(degrees=(-70, 70)),
                        v2.RandomAffine(degrees=(-15, 15), translate=(0.25, 0.25), scale=(0.7, 1.2), shear=(-15, 15, -15, 15)),
                        v2.RandomPerspective(distortion_scale=0.2, p=0.2),
                        v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.2), antialias=True),
                        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                        v2.RandomAutocontrast(p=0.2),
                        v2.RandomEqualize(p=0.2),
                        v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                    ])
                
                val_transform = transforms.Compose([
                    v2.ToTensor(),
                    v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
                    v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),

                    ])


                # trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=train_transform)
                # valset = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=val_transform)
                # test_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=val_transform)

                train_dataset = RSNADataset(csv_file="dataset/rsna_18/csv/final_dataset_wo_not_normal_cases.csv", data_folder="dataset/rsna_18", split="train", transform=train_transform)

                val_dataset = RSNADataset(csv_file="dataset/rsna_18/csv/final_dataset_wo_not_normal_cases.csv", data_folder="dataset/rsna_18", split="val", transform=val_transform)
                
                test_dataset = RSNADataset(csv_file="dataset/rsna_18/csv/final_dataset_wo_not_normal_cases.csv", data_folder="dataset/rsna_18", split="test", transform=val_transform)

                # train_size = int(0.9 * len(trainset))
                # val_size = len(trainset) - train_size
                
                # _, val_dataset = torch.utils.data.random_split(valset, [train_size, val_size])

                # train_dataset, _ = torch.utils.data.random_split(trainset, [train_size, val_size])
    

                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

                optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)

                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, verbose=True)

     
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
                    epochs=num_epoch,
                    save_dir=save_dir,
                )


                model = load_model(model, save_dir + "/best_model.pt")
                model.to(DEVICE)
                torch.compile(model)

                test_loss, test_acc = val_step(model, test_loader, loss, DEVICE)

                wandb.log({"test_loss": test_loss, "test_acc": test_acc})

                config["test_acc"] = test_acc
                config["test_loss"] = test_loss

                train_summary = {
                    "config": wandconf,
                    "results": results,
                }

                with open(save_dir + "/train_summary.json", "w") as f:
                    json.dump(train_summary, f, indent=4)

                plot_results(results, save_dir)

                #append to dataframe
                resultsexp.loc[len(resultsexp)] = [wandconf["LEARNING_RATE"], wandconf["NUM_EPOCHS"], wandconf["AUGMENTATION"], test_acc, test_loss]            

    resultsexp.to_csv(parent_dir + "/testresults.csv", index=True)

if __name__ == "__main__":
    main()


