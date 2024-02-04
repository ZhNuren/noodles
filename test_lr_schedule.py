from utils.utils import souping, greedy_souping
from models import get_model
from engine import trainer, val_step

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import tabulate

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

from utils.utils import EarlyStopper, get_dataset, cyclic_learning_rate, adjust_learning_rate, train, test, save_checkpoint, cyclic_learning_rate_v2
from torch.utils.data.dataset import Subset

import yaml
import json
import time
import os
import wandb
from utils.utils import load_model
import sys


def START_seed(start_seed=9):
    seed = start_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/fge.yaml', metavar='DIR', help='configs')

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
LR1 = config['LR1']
LR2 = config['LR2']
cycle = 5#config['CYCLE']
WD = config['WD']
MOMENTUM = config["MOMENTUM"]

# assert cycle % 2 == 0, 'Cycle length should be even'

print('>>>>>>>>>>>>>>>>>>>>>>>>>>', DATASET,'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


#if there is no test directory create it
if not os.path.exists("/l/users/santosh.sanjeev/model_soups/runs/test"):
    os.makedirs("/l/users/santosh.sanjeev/model_soups/runs/test")

#if there is no folder named after dataset inside test directory create it
if not os.path.exists(os.path.join("/l/users/santosh.sanjeev/model_soups/runs/test", "fge", DATASET+ '_3', MODEL)):
    os.makedirs(os.path.join("/l/users/santosh.sanjeev/model_soups/runs/test", "fge", DATASET + '_3', MODEL))

# test_save_path = os.path.join("/l/users/santosh.sanjeev/model_soups/runs/test", "fge", DATASET+ '_3', MODEL)
# # Open a text file for logging
# log_file_path = os.path.join("/l/users/santosh.sanjeev/model_soups/runs/test/fge", DATASET+ '_3', MODEL, "output_log.txt")
# log_file = open(log_file_path, "w")

# # Redirect the standard output to the log file
# sys.stdout = log_file


# train_loader, val_loader, test_loader = get_dataset(DATASET, PATHS, "Minimal", 
#                                                         PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK)

LOSS = config["LOSS"]
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


#### create dictionary with learning_rates as keys and only the model paths with heavy augmentations as the keys (the keys are list of keys)
DICTIONARY = {}

for idx, folder in enumerate(sorted(os.listdir(WEIGHT_PATH))):
    if folder.startswith('testresults'):
        continue
    model_path = os.path.join(WEIGHT_PATH, folder)
    #read config in train_summary.json
    train_summary = json.load(open(os.path.join(model_path, "train_summary.json"), 'r'))
    model_config = train_summary["config"]
    if model_config['AUGMENTATION'] == 'Heavy':
        if model_config['LEARNING_RATE'] in DICTIONARY.keys():# is in DICTIONARY.keys():
            DICTIONARY[model_config['LEARNING_RATE']].append(model_path)
        else:
            DICTIONARY[model_config['LEARNING_RATE']] = []
            DICTIONARY[model_config['LEARNING_RATE']].append(model_path)
# print(DICTIONARY)


for lr in DICTIONARY.keys():
    state_dicts = []
    val_results = []
    test_results = []
    val_models = [0]
    test_models = [0]

    if lr!= '1.00e-05':
        continue

    if not os.path.exists(f'/l/users/santosh.sanjeev/model_soups/runs/test/fge/{DATASET}_3/{MODEL}/{lr}'):
        os.makedirs(f'/l/users/santosh.sanjeev/model_soups/runs/test/fge/{DATASET}_3/{MODEL}/{lr}')
    save_dir = f'/l/users/santosh.sanjeev/model_soups/runs/test/fge/{DATASET}_3/{MODEL}/{lr}'

    print('SAVE_DIRRRRRRRRRRRR', save_dir)
    model_path = random.sample(DICTIONARY[lr], 1)[0]
    checkpoint = torch.load(os.path.join(model_path, "best_checkpoint.pth"), map_location=DEVICE)
    state_dicts.append(checkpoint['model'])
    model = get_model(model_config["MODEL"], num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint['model'])
    model = model.to(DEVICE)
    LR1 = float(lr) *1 #added
    LR2 = LR1 * 0.01 #added
    optimizer = torch.optim.AdamW(model.parameters(),lr=LR1,weight_decay=WD) #added
    optimizer.load_state_dict(checkpoint['optimizer'])
    train_loader, val_loader, test_loader = get_dataset(DATASET, PATHS, 'Heavy', PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK)
    val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(model, val_loader, train_loader, loss, device = DEVICE, classification = CLASSIFICATION)
    test_loss, test_acc, test_f1, test_recall, test_kappa, test_auc = val_step(model, test_loader, train_loader, loss, device = DEVICE, classification = CLASSIFICATION)

    print('BASE MODEL', val_acc, test_acc)
    val_results.append(val_acc)
    test_results.append(test_acc)
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'va_acc', 'te_acc', 'time']
    augmentation = 'Minimal'
    seed = 0
    aug_variable = 0
    seed_variable = 0
    
    

    epoch_list = []
    lr_list = []
    for epoch in range(1, 27):
        # time_ep = time.time()
        # print('epochhhhhhhhhhhhhhhhhhh', epoch)
        # if (epoch % cycle + 1) == cycle // 2:
        #     if (epoch-1) % 8 == 0:
        #         seed_variable = 0

        #     aug_list = ['Minimal', 'Medium', 'Heavy']
        #     see_list = [0, 1]

        #     if (epoch-1)%8 ==0:
        #         aug_variable+=1
        #     seed_variable +=1
        #     augmentation = aug_list[aug_variable-1]          
        #     seed = see_list[seed_variable-1]
        #     print(epoch, augmentation, seed)
        #     START_seed(start_seed = seed)
        #     train_loader, val_loader, test_loader = get_dataset(DATASET, PATHS, augmentation, PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK)

        time_ep = time.time()
        lr_schedule = cyclic_learning_rate_v2(epoch, cycle, LR1, LR2)
        num_iters = len(train_loader)
        model.train()
        for iter in range(num_iters):
        # for iter, (input, target) in enumerate(((train_loader))):
            if lr_schedule is not None:
                lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        epoch_list.append(epoch)
        lr_list.append(lr)
        print(epoch, lr)
    #     train_res = train(train_loader, model, optimizer, loss, lr_schedule=lr_schedule)
    #     # test_res = test(val_loader, model, loss)
    #     val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(model, val_loader, train_loader, loss, device = DEVICE, classification = CLASSIFICATION)
    #     test_loss, test_acc, test_f1, test_recall, test_kappa, test_auc = val_step(model, test_loader, train_loader, loss, device = DEVICE, classification = CLASSIFICATION)
    #     val_results.append(val_acc)
    #     test_results.append(test_acc)

    #     time_ep = time.time() - time_ep
    #     if (epoch % cycle + 1) == cycle // 2:
    #         print(epoch)
    #         save_checkpoint(
    #             save_dir,
    #             epoch,
    #             name='fge',
    #             model_state=model.state_dict(),
    #             optimizer_state=optimizer.state_dict()
    #         )
    #         state_dicts.append(model.state_dict())
    #         val_models.append(epoch)
    #         test_models.append(epoch)

    #     values = [epoch, lr_schedule(1.0), train_res['loss'], train_res['accuracy'], val_acc, test_acc, time_ep]
    #     table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.8f')
    #     print()
    #     if epoch % 40 == 0:
    #         table = table.split('\n')
    #         table = '\n'.join([table[1]] + table)
    #     else:
    #         table = table.split('\n')[2]
    #     print(epoch, val_acc, test_acc)
    # print(table)

    # print("Uniform souping ...")
    # alphal = [1 / len(state_dicts) for i in range(len(state_dicts))]
    # model = get_model(model_config["MODEL"], num_classes=NUM_CLASSES)
    # uniform_model = souping(model, state_dicts, alphal)
    # uniform_model.to(DEVICE)

    # uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc = val_step(uniform_model, test_loader, train_loader, loss, DEVICE, CLASSIFICATION)
    # print('UNIFORMMMMMMMMMMM', uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc)
    # #greedy

    # print("Greedy souping ...")
    # model = get_model(model_config["MODEL"], num_classes=NUM_CLASSES)
    # greedy_model, best_ingredients = greedy_souping(state_dicts, val_results, model_config["MODEL"], NUM_CLASSES, val_loader, train_loader, loss, DEVICE , CLASSIFICATION, val_sort_by, val_models =val_models)
    # greedy_model.to(DEVICE)
    # print('VAL INGREDIENTS',best_ingredients)
    # greedy_test_loss, greedy_test_acc, greedy_test_f1, greedy_test_recall, greedy_test_kappa, greedy_test_auc = val_step(greedy_model, test_loader, train_loader, loss, DEVICE, CLASSIFICATION)
    # print('GREEDYYYY VAL INGREDIENTS TEST', greedy_test_loss, greedy_test_acc, greedy_test_f1, greedy_test_recall, greedy_test_kappa, greedy_test_auc)


    # greedy_model_test, best_ingredients_test = greedy_souping(state_dicts, test_results, model_config["MODEL"], NUM_CLASSES, test_loader, train_loader, loss, DEVICE, CLASSIFICATION, test_sort_by, test_models)
    # print('GREEDYYYY TEST INGREDIENTS TEST', best_ingredients_test)
    # greedy_model_test.to(DEVICE)
    # greedy_test_loss_test, greedy_test_acc_test, greedy_test_f1_test, greedy_test_recall_test, greedy_test_kappa_test, greedy_test_auc_test = val_step(greedy_model_test, test_loader, train_loader, loss, DEVICE, CLASSIFICATION)

    # print('GREEDYYYY TEST', greedy_test_loss_test, greedy_test_acc_test, greedy_test_f1_test, greedy_test_recall_test, greedy_test_kappa_test, greedy_test_auc_test)
# log_file.close()
        
plt.plot(epoch_list, lr_list)

# Adding labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Your Graph Title')

# Set x-axis ticks to whole numbers
plt.xticks(range(1, len(epoch_list)+1))

# Save the figure without displaying
plt.savefig('var_lr_schedule_cycle_5_v3.png')
