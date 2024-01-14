import sys
sys.path.append('./')
from engine import trainer, val_step
from utils.utils import plot_results
from pretraining.models import get_model

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform
from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()
from timm.data.transforms import RandomResizedCropAndInterpolation
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import argparse

from utils.utils import EarlyStopper
from torch.utils.data.dataset import Subset
from pretraining.datasets import DRDataset, get_dataset

import yaml
import json
import time
import os
import wandb
from utils.utils import load_model


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/san_config_finetune.yaml', metavar='DIR', help='configs')
args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
print(config)
run = wandb.init(entity='biomed', project='model_soups', config=config)


wandconf = {
        "LEARNING_SCHEDULER": config["LEARNING_SCHEDULER"],
        "BATCH_SIZE": config["BATCH_SIZE"],
        "IMAGE_SIZE": config["IMAGE_SIZE"],
        "MODEL": config["MODEL"],
        "PRETRAINED": config["PRETRAINED"],
        "LOSS": config["LOSS"],
        "SAVE_DIR": config["SAVE_DIR"],
        "RESUME_PATH": config["RESUME_PATH"]

}

LEARNING_SCHEDULER = config["LEARNING_SCHEDULER"]
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])
NUM_CLASSES = int(config["NUM_CLASSES"])
LINEAR_PROBING = config["LINEAR_PROBING"]
# PROBING_EPOCHS = int(config["PROBING_EPOCHS"])
PATIENCE = int(config["PATIENCE"])
SAVE_DIR = str(config["SAVE_DIR"])
LR_RATE_LIST = config["LR_RATE_LIST"]
PATHS = config["PATH"]
DATASET = config["DATASET"]
LOSS = config["LOSS"]
IMAGE_SIZE = int(config["IMAGE_SIZE"])
MODEL = config["MODEL"]
PRETRAINED = config["PRETRAINED"]
CUDA_DEVICE = int(config["CUDA_DEVICE"])
NUM_WORKERS = int(config["NUM_WORKERS"])
PRETRAINING = str(config["PRETRAINING"])
TASK = config["TASK"]
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else 'cpu')
INITIALISATION = str(config["INITIALISATION"])
RESUME_PATH = str(config["RESUME_PATH"])

print(f"Using {DEVICE} device")


def START_seed(start_seed=9):
    seed = start_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 


def main():
    global RESUME_PATH
    START_seed()
    if RESUME_PATH == "":
        run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        #create folder for this run in runs folder

        os.mkdir(SAVE_DIR + run_id)

        parent_dir = SAVE_DIR + run_id
    else:
        parent_dir = RESUME_PATH
    
    
    
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
    resultsexp = pd.DataFrame(columns=["lr_rate", "test_acc", "test_loss", "test_f1", "test_kappa"])
    


    print(LR_RATE_LIST)
    for lr_rate in LR_RATE_LIST:
        lr_rate = float(lr_rate)
        num_epoch = NUM_EPOCHS

        wandconf["LEARNING_RATE"] = "{:.2e}".format(lr_rate)
        wandconf["NUM_EPOCHS"] = num_epoch
        print(lr_rate, num_epoch,)                
        

        model = get_model(MODEL, TASK, PRETRAINED, num_classes=NUM_CLASSES)
        model.to(DEVICE)
        torch.compile(model)

        run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
        os.mkdir(parent_dir + "/hparam_" + run_id)
        save_dir = parent_dir + "/hparam_" + run_id


        train_loader, val_loader, test_loader = get_dataset(DATASET, PATHS, PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)

        if LEARNING_SCHEDULER == "CosineAnnealingLR":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, verbose=True)
        elif LEARNING_SCHEDULER == "CyclicLR":
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = lr_rate, max_lr = lr_rate * 0.01, cycle_momentum=False)

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

        checkpoint = torch.load(save_dir + "/best_checkpoint.pth")
        model.load_state_dict(checkpoint['model'])
        model.to(DEVICE)
        torch.compile(model)

        test_loss, test_acc, test_f1, test_recall,test_kappa = val_step(model, test_loader, loss, DEVICE)

        config["test_acc"] = test_acc
        config["test_loss"] = test_loss
        config["test_f1"] = test_f1
        config["test_recall"] = test_recall
        config["test_kappa"] = test_kappa
        

        wandb.log({"test_loss": test_loss, "test_acc": test_acc, "test_F1":test_f1, "test_recall":test_recall, "test_kappa":test_kappa})

        train_summary = {
            "config": config,
            "results": results,
        }


        with open(save_dir + "/train_summary.json", "w") as f:
            json.dump(train_summary, f, indent=4)

        plot_results(results, save_dir)

        #append to dataframe
        resultsexp.loc[len(resultsexp)] = [wandconf["LEARNING_RATE"], test_acc, test_loss, test_f1, test_recall, test_kappa]            

    resultsexp.to_csv(parent_dir + "/testresults.csv", index=True)

if __name__ == "__main__":
    main()


