from engine import val_step
from utils import souping, greedy_souping
from models import get_model

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

from dataset import RSNADataset, HAM10000Dataset, AptosDataset
from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform
from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()
from timm.data.transforms import RandomResizedCropAndInterpolation
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import argparse

from utils import EarlyStopper, get_dataset
from torch.utils.data.dataset import Subset

import yaml
import json
import time
import os
import wandb
from utils import load_model



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/test_config.yaml', metavar='DIR', help='configs')
args = parser.parse_args()


config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

DATASET = config["DATASET"]
CUDA_DEVICE = int(config["CUDA_DEVICE"])
NUM_WORKERS = int(config["NUM_WORKERS"])
PRETRAINING = str(config["PRETRAINING"])
WEIGHT_PATH = config["WEIGHT_PATH"]
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_CLASSES = int(config["NUM_CLASSES"])
IMAGE_SIZE = int(config["IMAGE_SIZE"])


PATHS = config["PATH"]

DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else 'cpu')

state_dicts = []
val_results = []
test_results = []

_, val_loader, test_loader = get_dataset(DATASET, PATHS, "Minimal", 
                                                        PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS)

#for each folder inside WEIGHT_PATH
for idx, folder in enumerate(os.listdir(WEIGHT_PATH)):
    model_path = os.path.join(WEIGHT_PATH, folder)
    #read config in train_summary.json
    train_summary = json.load(open(os.path.join(model_path, "train_summary.json"), 'r'))
    model_config = train_summary["config"]
    
    #load model
    checkpoint = torch.load(os.path.join(model_path, "best_checkpoint.pth"), map_location=DEVICE)
    state_dicts.append(checkpoint['model'])
    model = get_model(model_config["MODEL"], num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint['model'])
    model = model.to(DEVICE)
    #load loss function
    LOSS = model_config["LOSS"]
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
    
    test_loss, test_acc, test_f1, test_recall = val_step(model, test_loader, loss, DEVICE)
    val_loss, val_acc, val_f1, val_recall = val_step(model, val_loader, loss, DEVICE)

    val_results.append({'Model Name': idx,
                                    'Val Accuracy': val_acc,
                                    'Val F1': val_f1,
                                    'Val Recall': val_recall,
                                    'Augmentation': model_config["AUGMENTATION"],
                                    'Learning Rate': model_config["LEARNING_RATE"],
                                    'SEED': model_config["SEED"]})
    test_results.append({'Model Name': idx,
                                    'Test Accuracy': test_acc,
                                    'Test F1': test_f1,
                                    'Test Recall': test_recall,
                                    'Augmentation': model_config["AUGMENTATION"],
                                    'Learning Rate': model_config["LEARNING_RATE"],
                                    'SEED': model_config["SEED"]})
    
results_test_df = pd.DataFrame(test_results)
results_val_df = pd.DataFrame(val_results)

val_copy = results_val_df.copy()
sorted_val = val_copy.sort_values(by= 'Val F1',ascending=False)

test_copy = results_test_df.copy()
sorted_test = test_copy.sort_values(by= 'Test F1',ascending=False)

print(f"Best model val F1: {sorted_val.iloc[0]['Val F1']}")
print(f"Second best model val F1: {sorted_val.iloc[1]['Val F1']}")
print(f"Worst model val F1: {sorted_val.iloc[-1]['Val F1']}")

#UNIFORM
print("Unifrom souping ...")
alphal = [1 / len(state_dicts) for i in range(len(state_dicts))]

model = get_model(model_config["MODEL"], num_classes=NUM_CLASSES)

uniform_model = souping(model, state_dicts, alphal)

uniform_model.to(DEVICE)

uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall = val_step(uniform_model, test_loader, loss, DEVICE)


# uniform_val_loss, uniform_val_acc, uniform_val_f1, uniform_val_recall = val_step(uniform_model, val_loader, loss, DEVICE)

#greedy
print("Greedy souping ...")
model = get_model(model_config["MODEL"], num_classes=NUM_CLASSES)

val_f1 = list(results_val_df['Val F1'])
greedy_model = greedy_souping(state_dicts, val_f1, model_config["MODEL"], NUM_CLASSES, val_loader, loss, DEVICE)

greedy_model.to(DEVICE)

# greedy_val_loss, greedy_val_acc, greedy_val_f1, greedy_val_recall = val_step(greedy_model, val_loader, loss, DEVICE)
#TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTESTTTTTTTTTTTTTTTTTTTTTTTTT


#test all the 48 models - done


#take the unifrom model from above and test it

greedy_test_loss, greedy_test_acc, greedy_test_f1, greedy_test_recall = val_step(greedy_model, test_loader, loss, DEVICE)

# take the greedy souped model and test it

#create following pandas table columns(F1, Accuracy, Recall, Augmentation, Learning Rate, SEED)

print("Creating table ...")
table = pd.DataFrame(columns=['Model Name', 'Test Accuracy', 'Test F1', 'Test Recall', 'Augmentation', 'Learning Rate', 'SEED'])

#get best, second best and worst model from sorted_test and add them to the table
best_model = sorted_test.iloc[0]
second_best_model = sorted_test.iloc[1]
worst_model = sorted_test.iloc[-1]

#rename model name in best, second best and worst model
best_model['Model Name'] = f"Best 1: {best_model['Model Name']}"
second_best_model['Model Name'] = f"Best 2: {second_best_model['Model Name']}"
worst_model['Model Name'] = f"Worst: {worst_model['Model Name']}"

table.loc[0] = best_model
table.loc[1] = second_best_model
table.loc[2] = worst_model

#add uniform and greedy to the table
table.loc[3] = {'Model Name': 'Uniform',
                                    'Test Accuracy': uniform_test_acc,
                                    'Test F1': uniform_test_f1,
                                    'Test Recall': uniform_test_recall,
                                    'Augmentation': 'None',
                                    'Learning Rate': 'None',
                                    'SEED': 'None'}

table.loc[4] = {'Model Name': 'Greedy',
                                    'Test Accuracy': greedy_test_acc,
                                    'Test F1': greedy_test_f1,
                                    'Test Recall': greedy_test_recall,
                                    'Augmentation': 'None',
                                    'Learning Rate': 'None',
                                    'SEED': 'None'}

#save the table to csv without index
table.to_csv("test_results.csv", index=False)
print(table.to_markdown())
print("Table saved to test_results.csv")





#OUT ALL THE TEST RESULTS AND CORRESPONDING HYPERPARAMETERS to the table
#1st best, 2nd best, worst, uniform and greedy