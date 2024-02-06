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
import tabulate
import utils.dnn_utils as dnn_utils




parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/plane.yaml', metavar='DIR', help='configs')
parser.add_argument('--grid_points', type=int, default=21, metavar='N',
                    help='number of points in the grid (default: 21)')
parser.add_argument('--margin_left', type=float, default=0.2, metavar='M',
                    help='left margin (default: 0.2)')
parser.add_argument('--margin_right', type=float, default=0.2, metavar='M',
                    help='right margin (default: 0.2)')
parser.add_argument('--margin_bottom', type=float, default=0.2, metavar='M',
                    help='bottom margin (default: 0.)')
parser.add_argument('--margin_top', type=float, default=0.2, metavar='M',
                    help='top margin (default: 0.2)')

parser.add_argument('--curve_points', type=int, default=61, metavar='N',)
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')
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
CLASSIFICATION = str(config["CLASSIFICATION"])
val_sort_by = str(config["val_sort_by"])
test_sort_by = str(config["test_sort_by"])
TASK = str(config["TASK"])
PATHS = config["PATH"]
MODEL = config["MODEL"]
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else 'cpu')

SAVE_PATH = str(config['SAVE_PATH'])
print('>>>>>>>>>>>>>>>>>>>>>>>>>>', DATASET,'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
state_dicts = []
val_results = []
test_results = []

if not os.path.exists(f'/home/santosh.sanjeev/Projects/model-soups/noodles/planes/{DATASET}/{MODEL}/{SAVE_PATH}'):
    os.makedirs(f'/home/santosh.sanjeev/Projects/model-soups/noodles/planes/{DATASET}/{MODEL}/{SAVE_PATH}')
SAVE_PATH = '/home/santosh.sanjeev/Projects/model-soups/noodles/planes/{DATASET}/{MODEL}/{SAVE_PATH}'
print(SAVE_PATH)
train_loader, val_loader, test_loader = get_dataset(DATASET, PATHS, "Minimal", 
                                                        PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK)


criterion = torch.nn.CrossEntropyLoss()
regularizer = dnn_utils.l2_regularizer(args.wd)

paths = []
for idx, folder in enumerate(sorted(os.listdir(WEIGHT_PATH))):
    if folder.startswith('testresults'):
        continue
    model_path = os.path.join(WEIGHT_PATH, folder, "best_checkpoint.pth")

    paths.append(model_path)
print(len(paths))
initial_model_path = '/home/santosh.sanjeev/Projects/model-soups/noodles/runs/san_final_hyp_models/san-initial_models/aptos_final_hyp/deitS_imagenet/2024-01-10_02-40-01/best_checkpoint.pth'

MODEL1 = paths[22]#initial_model_path
MODEL2 = paths[28]
MODEL3 = paths[23]


def get_xy(point, origin, vector_x, vector_y):
    return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])

checkpoint1 = torch.load(MODEL1, map_location=DEVICE)
model1 = get_model(MODEL, num_classes=NUM_CLASSES)
model1.load_state_dict(checkpoint1['model'])
model1 = model1.to(DEVICE)
test_res = dnn_utils.test(val_loader, model1, criterion, regularizer)
print('111111111',test_res['accuracy'])
# val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(model1, val_loader, train_loader, criterion, device = DEVICE, classification = CLASSIFICATION)
checkpoint2 = torch.load(MODEL2, map_location=DEVICE)
model2 = get_model(MODEL, num_classes=NUM_CLASSES)
model2.load_state_dict(checkpoint2['model'])
model2 = model2.to(DEVICE)
test_res = dnn_utils.test(val_loader, model2, criterion, regularizer)
print('22222222',test_res['accuracy'])

checkpoint3 = torch.load(MODEL3, map_location=DEVICE)
model3 = get_model(MODEL, num_classes=NUM_CLASSES)
model3.load_state_dict(checkpoint3['model'])
model3 = model3.to(DEVICE)
test_res = dnn_utils.test(val_loader, model3, criterion, regularizer)
print('333333333',test_res['accuracy'])



w = list()
#curve_parameters = []#list(curve_model.net.parameters())
curve_parameters = [model1.parameters(),model2.parameters(),model3.parameters()]#list(curve_model.net.parameters())

for i in range(3):
    w.append(np.concatenate([
        p.data.cpu().numpy().ravel() for p in curve_parameters[i]#[i::args.num_bends]
    ]))

print('Weight space dimensionality: %d' % w[0].shape[0])

u = w[2] - w[0]
dx = np.linalg.norm(u)
u /= dx

v = w[1] - w[0]
v -= np.dot(u, v) * u
dy = np.linalg.norm(v)
v /= dy

bend_coordinates = np.stack(get_xy(p, w[0], u, v) for p in w)
ts = np.linspace(0.0, 1.0, args.curve_points)
# curve_coordinates = []
# for t in np.linspace(0.0, 1.0, args.curve_points):
#     weights = curve_model.weights(torch.Tensor([t]).cuda())
#     curve_coordinates.append(get_xy(weights, w[0], u, v))
# curve_coordinates = np.stack(curve_coordinates)

G = args.grid_points
alphas = np.linspace(0.0 - args.margin_left, 1.0 + args.margin_right, G)
betas = np.linspace(0.0 - args.margin_bottom, 1.0 + args.margin_top, G)
# print('dx',- args.margin_left*dx,(1.0 + args.margin_right)*dx,'dy',- args.margin_bottom*dx,(1.0 + args.margin_top)*dy)
print(bend_coordinates)
tr_loss = np.zeros((G, G))
tr_nll = np.zeros((G, G))
tr_acc = np.zeros((G, G))
tr_err = np.zeros((G, G))

te_loss = np.zeros((G, G))
te_nll = np.zeros((G, G))
te_acc = np.zeros((G, G))
te_err = np.zeros((G, G))

grid = np.zeros((G, G, 2))

base_model = get_model(MODEL, num_classes=NUM_CLASSES)
base_model.to(DEVICE)

columns = ['X', 'Y', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']

for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        p = w[0] + alpha * dx * u + beta * dy * v

        offset = 0
        for parameter in base_model.parameters():
            size = np.prod(parameter.size())
            value = p[offset:offset+size].reshape(parameter.size())
            parameter.data.copy_(torch.from_numpy(value))
            offset += size

        dnn_utils.update_bn(train_loader, base_model)

        tr_res = dnn_utils.test(train_loader, base_model, criterion, regularizer)
        te_res = dnn_utils.test(val_loader, base_model, criterion, regularizer)

        tr_loss_v, tr_nll_v, tr_acc_v = tr_res['loss'], tr_res['nll'], tr_res['accuracy']
        te_loss_v, te_nll_v, te_acc_v = te_res['loss'], te_res['nll'], te_res['accuracy']

        c = get_xy(p, w[0], u, v)
        grid[i, j] = [alpha * dx, beta * dy]

        tr_loss[i, j] = tr_loss_v
        tr_nll[i, j] = tr_nll_v
        tr_acc[i, j] = tr_acc_v
        tr_err[i, j] = 100.0 - tr_acc[i, j]

        te_loss[i, j] = te_loss_v
        te_nll[i, j] = te_nll_v
        te_acc[i, j] = te_acc_v
        te_err[i, j] = 100.0 - te_acc[i, j]

        values = [
            grid[i, j, 0], grid[i, j, 1], tr_loss[i, j], tr_nll[i, j], tr_err[i, j],
            te_nll[i, j], te_err[i, j]
        ]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
        if j == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

np.savez(
    os.path.join(SAVE_PATH, 'plane.npz'),
    ts=ts,
    bend_coordinates=bend_coordinates,
    # curve_coordinates=curve_coordinates,
    alphas=alphas,
    betas=betas,
    grid=grid,
    tr_loss=tr_loss,
    tr_acc=tr_acc,
    tr_nll=tr_nll,
    tr_err=tr_err,
    te_loss=te_loss,
    te_acc=te_acc,
    te_nll=te_nll,
    te_err=te_err
)
