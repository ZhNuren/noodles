from engine import val_step
from utils.utils import souping, greedy_souping
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

from utils.utils import EarlyStopper, get_dataset
from torch.utils.data.dataset import Subset

import yaml
import json
import time
import os
import wandb
from utils.utils import load_model
import sys


WEIGHT_PATH_v2 = '/l/users/santosh.sanjeev/model_soups/runs/san_final_hyp_models/san-finetune/aptos_final_hyp/deitS_imagenet/2024-01-10_02-53-56/'
initial_val_df = pd.read_csv('/home/santosh.sanjeev/Projects/model-soups/noodles/test/APTOS/DeiT-S/VAL_RESULTS.csv')
initial_test_df = pd.read_csv('/home/santosh.sanjeev/Projects/model-soups/noodles/test/APTOS/DeiT-S/TEST_RESULTS.csv')
NUM_CLASSES = 5
MODEL = 'DeiT-S'
DEVICE = 'cuda'
val_sort_by = 'Val F1'
initial_model_path = '/home/santosh.sanjeev/Projects/model-soups/noodles/runs/san_final_hyp_models/san-initial_models/aptos_final_hyp/deitS_imagenet/2024-01-10_02-40-01/best_checkpoint.pth'


state_dicts = []
for idx, folder in enumerate(sorted(os.listdir(WEIGHT_PATH_v2))):
    print(idx)
    if folder.startswith('testresults'):
        continue
    model_path = os.path.join(WEIGHT_PATH_v2, folder)
    train_summary = json.load(open(os.path.join(model_path, "train_summary.json"), 'r'))
    model_config = train_summary["config"]

    checkpoint = torch.load(os.path.join(model_path, "best_checkpoint.pth"), map_location=DEVICE)
    state_dicts.append(checkpoint['model'])
    if idx==5:
        break

alphal = [1 / len(state_dicts) for i in range(len(state_dicts))]
model = get_model(MODEL, num_classes=NUM_CLASSES)
uniform_model = souping(model, state_dicts, alphal)
uniform_model.to(DEVICE)



angle_list = []
distance_list = []
for index, row in initial_val_df.iterrows():
    model_id = initial_val_df.iloc[index]['Model Name']
    # model = get_model(MODEL, num_classes=NUM_CLASSES)
    # model = model.to(DEVICE)
    model1 = get_model(MODEL, num_classes=NUM_CLASSES)
    model1.load_state_dict(torch.load(initial_model_path,map_location='cuda')['model'])
    model1 = model1.to(DEVICE)
    
    model2 = uniform_model #get_model(MODEL, num_classes=NUM_CLASSES)

    model3 = get_model(MODEL, num_classes=NUM_CLASSES)
    model3 = model3.to(DEVICE)

    w = list()
    curve_parameters = [model1.parameters(), model2.parameters(), model3.parameters()]
    for i in range(3):
        w.append(np.concatenate([
            p.data.cpu().numpy().ravel() for p in curve_parameters[i]
        ]))
    breakpoint()
    u = w[2] - w[0]
    dx = np.linalg.norm(u)
    u /= dx
    v = w[1] - w[0]
    # v -= np.dot(u, v) * u
    dy = np.linalg.norm(v)
    v /= dy
    # bend_coordinates = np.stack(get_xy(p, w[0], u, v) for p in w)
    angle, distance = angle_distance_between_vectors(u,v)
    print(angle, distance)
    angle_list.append(angle)
    distance_list.append(distance)
    # model_data.append((model_id, angle, initial_val_df.iloc[index]['Learning Rate']))
#     print(model_id, angle, initial_val_df.iloc[index]['Learning Rate'])
# # Sort the model IDs based on angles in descending order
# sorted_model_data = sorted(model_data, key=lambda x: x[1], reverse=True)

# # Extract sorted model IDs and angles
# sorted_model_ids = [model_id for model_id, angle, LRv in sorted_model_data]
# sorted_angles = [angle for model_id, angle, LRv in sorted_model_data]
# sorted_LRs = [LRv for model_id, angle, LRv in sorted_model_data]
# print("Sorted Model IDs based on angle:")
# print(sorted_model_ids)
# print("Corresponding Angles:")
# print(sorted_angles)
# print("Corresponding LR:")
# print(sorted_LRs)
