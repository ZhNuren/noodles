# from engine import val_step
# from utils.utils import souping, greedy_souping
# from models import get_model

# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import torchvision

# from dataset import RSNADataset, HAM10000Dataset, AptosDataset
# from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform
# from torchvision.transforms import v2
# torchvision.disable_beta_transforms_warning()
# from timm.data.transforms import RandomResizedCropAndInterpolation
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# import pandas as pd
# import argparse

# from utils.utils import EarlyStopper, get_dataset
# from torch.utils.data.dataset import Subset

# import yaml
# import json
# import time
# import os
# import wandb
# from utils.utils import load_model
# import sys

# DATASET = 'HAM'
# MODEL = 'DeiT-S'
# NUM_CLASSES = 100
# #if there is no folder named after dataset inside test directory create it
# if not os.path.exists(os.path.join("test", "fge", DATASET, MODEL)):
#     os.makedirs(os.path.join("test", "fge", DATASET, MODEL))

# test_save_path = os.path.join("test", "fge", DATASET, MODEL)

# # # Open a text file for logging
# log_file_path = os.path.join("test", "fge", DATASET, MODEL, "output_log.txt")
# log_file = open(log_file_path, "w")

# # Redirect the standard output to the log file
# sys.stdout = log_file



# state_dicts, val_results, test_results = [], [], []
# loss = torch.nn.CrossEntropyLoss()
# train_loader, val_loader, test_loader = get_dataset(DATASET, './dataset/cifar""./dataset/cifar/cifar1098_idxs.npy', "Minimal", 'ImageNet', 224, 128, 16, 'Classification')

# WEIGHT_PATH = f'/l/users/santosh.sanjeev/model_soups/runs/test/fge/{DATASET}/DeiT-S/'
# for idx, folder in enumerate(sorted(os.listdir(WEIGHT_PATH))):
#     print(folder)
#     if folder in ['1.00e-05', '5.00e-06']:
#         continue
#     if folder.startswith('output'):
#         continue
#     folder_path = os.path.join(WEIGHT_PATH, folder)
#     for idx, model in enumerate(sorted(os.listdir(folder_path))):
#         if model.endswith('RESULTS.csv'):
#             continue
#         model_path = os.path.join(folder_path, model)
#         #read config in train_summary.json
#         print(model_path)

#         checkpoint = torch.load(model_path, map_location='cuda')
#         if '.pth' in model_path:
#             state_dicts.append(checkpoint['model'])
#             model = get_model('DeiT-S', num_classes=NUM_CLASSES)
#             model.load_state_dict(checkpoint['model'])
#             model = model.to('cuda')
#         else:        
#             state_dicts.append(checkpoint['model_state'])
#             model = get_model('DeiT-S', num_classes=NUM_CLASSES)
#             model.load_state_dict(checkpoint['model_state'])
#             model = model.to('cuda')
#         test_loss, test_acc, test_f1, test_recall, test_kappa, test_auc = val_step(model, test_loader, train_loader, loss, device = 'cuda', classification = 'MultiClass')
#         val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(model, val_loader, train_loader, loss, device = 'cuda', classification = 'MultiClass')

#         val_results.append({'Model Name': idx,
#                                         'Val Accuracy': val_acc,
#                                         'Val F1': val_f1,
#                                         'Val Recall': val_recall,
#                                         'Val Kappa': val_kappa,
#                                         'Val AUC': val_auc,})
#         test_results.append({'Model Name': idx,
#                                         'Test Accuracy': test_acc,
#                                         'Test F1': test_f1,
#                                         'Test Recall': test_recall,
#                                         'Test Kappa': test_kappa,
#                                         'Test AUC': test_auc,})
        

# results_test_df = pd.DataFrame(test_results)
# results_val_df = pd.DataFrame(val_results)
# print(results_val_df)
# print(results_test_df)

# # val_copy = results_val_df.copy()
# # sorted_val = val_copy.sort_values(by= 'Val Accuacy',ascending=False)
# # sorted_val.to_csv(os.path.join(WEIGHT_PATH, "VAL_RESULTS.csv"), index=False)

# # test_copy = results_test_df.copy()
# # sorted_test = test_copy.sort_values(by= 'Test Accuracy',ascending=False)
# # sorted_test.to_csv(os.path.join(WEIGHT_PATH, "TEST_RESULTS.csv"), index=False)


# print("Uniform souping ...")
# alphal = [1 / len(state_dicts) for i in range(len(state_dicts))]
# model = get_model('DeiT-S', num_classes=NUM_CLASSES)
# uniform_model = souping(model, state_dicts, alphal)
# uniform_model.to('cuda')

# uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc = val_step(uniform_model, test_loader, train_loader, loss, 'cuda', 'MultiClass')
# print(uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc )
# #greedy

# print("Greedy souping ...")
# model = get_model('DeiT-S', num_classes=NUM_CLASSES)

# val_results = list(results_val_df['Val Accuracy'])
# val_models = list(results_val_df["Model Name"])
# greedy_model, best_ingredients = greedy_souping(state_dicts, val_results, 'DeiT-S', NUM_CLASSES, val_loader,train_loader, loss, 'cuda' , 'MultiClass', 'Val Accuracy', val_models =val_models)
# greedy_model.to('cuda')
# print('VAL INGREDIENTS',best_ingredients)
# greedy_test_loss, greedy_test_acc, greedy_test_f1, greedy_test_recall, greedy_test_kappa, greedy_test_auc = val_step(greedy_model, test_loader, train_loader, loss, 'cuda', 'MultiClass')
# print(greedy_test_loss, greedy_test_acc, greedy_test_f1, greedy_test_recall, greedy_test_kappa, greedy_test_auc)


# test_results = list(results_test_df['Test Accuracy'])
# test_models = list(results_test_df["Model Name"])
# greedy_model_test, best_ingredients_test = greedy_souping(state_dicts, test_results, 'DeiT-S', NUM_CLASSES, test_loader, train_loader, loss, 'cuda', 'MultiClass', 'Test Accuracy', test_models)
# print('TEST INGREDIENTS', best_ingredients_test)
# greedy_model_test.to('cuda')
# greedy_test_loss_test, greedy_test_acc_test, greedy_test_f1_test, greedy_test_recall_test, greedy_test_kappa_test, greedy_test_auc_test = val_step(greedy_model_test, test_loader, train_loader, loss, 'cuda', 'MultiClass')
# print(greedy_test_loss_test, greedy_test_acc_test, greedy_test_f1_test, greedy_test_recall_test, greedy_test_kappa_test, greedy_test_auc_test)

# log_file.close()


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



DATASET = 'Cifar100'
MODEL = 'DeiT-S'
NUM_CLASSES = 100
PATH = './dataset/cifar""./dataset/cifar/cifar1098_idxs.npy'#./dataset/224_data/DR/aptos/""./dataset/224_data/DR/aptos/aptos_dataset_splits.csv'#'/dataset/ISIC/ISIC2018_Task3_Training_Input/""./dataset/ISIC/ISIC2018_Task3_Validation_Input/""./dataset/ISIC/ISIC2018_Task3_Test_Input/""./dataset/ISIC/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv""./dataset/ISIC/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv""./dataset/ISIC/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv'
test_sort_by = 'Test Accuracy'
val_sort_by = 'Val Accuracy'
# if there is no folder named after dataset inside test directory create it
if not os.path.exists(os.path.join("/l/users/santosh.sanjeev/model_soups/runs/test/", "fge", DATASET, MODEL)):
    os.makedirs(os.path.join("/l/users/santosh.sanjeev/model_soups/runs/test/", "fge", DATASET, MODEL))

test_save_path = os.path.join("/l/users/santosh.sanjeev/model_soups/runs/test/", "fge", DATASET, MODEL)
print(test_save_path)
# # Open a text file for logging
log_file_path = os.path.join(test_save_path, "output_log_souping.txt")
print(log_file_path)
log_file = open(log_file_path, "w")

# Redirect the standard output to the log file
sys.stdout = log_file



state_dicts, val_results, test_results, val_dfs, test_dfs = [], [], [], [], []
loss = torch.nn.CrossEntropyLoss()
train_loader, val_loader, test_loader = get_dataset(DATASET, PATH, "Minimal", 'ImageNet', 224, 128, 16, 'Classification')
WEIGHT_PATH = f'/l/users/santosh.sanjeev/model_soups/runs/test/fge/{DATASET}/{MODEL}/'
for idx, folder in enumerate(sorted(os.listdir(WEIGHT_PATH))):
    if folder.startswith('output'):
        continue

    val_csv = os.path.join(WEIGHT_PATH, folder, 'VAL_RESULTS.csv')
    val_df = pd.read_csv(val_csv)
    for index, row in val_df.iterrows():
        for idx, wt in enumerate(sorted(os.listdir(os.path.join(WEIGHT_PATH, folder)))):
            if str(int(row['Model Name']))==0:
                if '--1' in wt or '-0' in wt:
                    model_path = os.path.join(WEIGHT_PATH, folder,wt)
            else:
                if str('-')+ str(int(row['Model Name'])) + str('.pt') in wt:
                    model_path = os.path.join(WEIGHT_PATH, folder,wt)
        val_df.at[index, 'LR'] = folder
        val_df.at[index, 'model_path'] = model_path

    
    # print(val_df)


    val_dfs.append(val_df)

# Concatenate all DataFrames into one
final_val_df = pd.concat(val_dfs)

final_val_df = final_val_df.sort_values(by=val_sort_by, ascending=False)
final_val_df.index = range(1, len(final_val_df) + 1)




for idx, folder in enumerate(sorted(os.listdir(WEIGHT_PATH))):
    if folder.startswith('output'):
        continue

    test_csv = os.path.join(WEIGHT_PATH, folder, 'TEST_RESULTS.csv')
    test_df = pd.read_csv(test_csv)
    for index, row in test_df.iterrows():
        for idx, wt in enumerate(sorted(os.listdir(os.path.join(WEIGHT_PATH, folder)))):
            if str(int(row['Model Name']))==0:
                if '--1' in wt or '-0' in wt:
                    model_path = os.path.join(WEIGHT_PATH, folder,wt)
            else:
                if str('-')+ str(int(row['Model Name'])) + str('.pt') in wt:
                    model_path = os.path.join(WEIGHT_PATH, folder,wt)
        test_df.at[index, 'LR'] = folder
        test_df.at[index, 'model_path'] = model_path
    test_dfs.append(test_df)

final_test_df = pd.concat(test_dfs)
final_test_df = final_test_df.sort_values(by=test_sort_by, ascending=False)
final_test_df.index = range(1, len(final_test_df) + 1)



# print(final_df)

#     for idx, model in enumerate(sorted(os.listdir(folder_path))):
#         if model.endswith('RESULTS.csv'):
#             continue
#         model_path = os.path.join(folder_path, model)
#         #read config in train_summary.json
#         print(model_path)

#         checkpoint = torch.load(model_path, map_location='cuda')
#         if '.pth' in model_path:
#             state_dicts.append(checkpoint['model'])
#             model = get_model('DeiT-S', num_classes=NUM_CLASSES)
#             model.load_state_dict(checkpoint['model'])
#             model = model.to('cuda')
#         else:        
#             state_dicts.append(checkpoint['model_state'])
#             model = get_model('DeiT-S', num_classes=NUM_CLASSES)
#             model.load_state_dict(checkpoint['model_state'])
#             model = model.to('cuda')
#         test_loss, test_acc, test_f1, test_recall, test_kappa, test_auc = val_step(model, test_loader, train_loader, loss, device = 'cuda', classification = 'MultiClass')
#         val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(model, val_loader, train_loader, loss, device = 'cuda', classification = 'MultiClass')

#         val_results.append({'Model Name': idx,
#                                         'Val Accuracy': val_acc,
#                                         'Val F1': val_f1,
#                                         'Val Recall': val_recall,
#                                         'Val Kappa': val_kappa,
#                                         'Val AUC': val_auc,})
#         test_results.append({'Model Name': idx,
#                                         'Test Accuracy': test_acc,
#                                         'Test F1': test_f1,
#                                         'Test Recall': test_recall,
#                                         'Test Kappa': test_kappa,
#                                         'Test AUC': test_auc,})
        

# results_test_df = pd.DataFrame(test_results)
# results_val_df = pd.DataFrame(val_results)
# print(results_val_df)
# print(results_test_df)

# # val_copy = results_val_df.copy()
# # sorted_val = val_copy.sort_values(by= 'Val Accuacy',ascending=False)
# # sorted_val.to_csv(os.path.join(WEIGHT_PATH, "VAL_RESULTS.csv"), index=False)

# # test_copy = results_test_df.copy()
# # sorted_test = test_copy.sort_values(by= 'Test Accuracy',ascending=False)
# # sorted_test.to_csv(os.path.join(WEIGHT_PATH, "TEST_RESULTS.csv"), index=False)


# print("Uniform souping ...")
# alphal = [1 / len(state_dicts) for i in range(len(state_dicts))]
# model = get_model('DeiT-S', num_classes=NUM_CLASSES)
# uniform_model = souping(model, state_dicts, alphal)
# uniform_model.to('cuda')

# uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc = val_step(uniform_model, test_loader, train_loader, loss, 'cuda', 'MultiClass')
# print(uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc )
# #greedy

# print("Greedy souping ...")
# model = get_model('DeiT-S', num_classes=NUM_CLASSES)

# val_results = list(results_val_df['Val Accuracy'])
# val_models = list(results_val_df["Model Name"])
# greedy_model, best_ingredients = greedy_souping(state_dicts, val_results, 'DeiT-S', NUM_CLASSES, val_loader,train_loader, loss, 'cuda' , 'MultiClass', 'Val Accuracy', val_models =val_models)
# greedy_model.to('cuda')
# print('VAL INGREDIENTS',best_ingredients)
# greedy_test_loss, greedy_test_acc, greedy_test_f1, greedy_test_recall, greedy_test_kappa, greedy_test_auc = val_step(greedy_model, test_loader, train_loader, loss, 'cuda', 'MultiClass')
# print(greedy_test_loss, greedy_test_acc, greedy_test_f1, greedy_test_recall, greedy_test_kappa, greedy_test_auc)


# test_results = list(results_test_df['Test Accuracy'])
# test_models = list(results_test_df["Model Name"])
# greedy_model_test, best_ingredients_test = greedy_souping(state_dicts, test_results, 'DeiT-S', NUM_CLASSES, test_loader, train_loader, loss, 'cuda', 'MultiClass', 'Test Accuracy', test_models)
# print('TEST INGREDIENTS', best_ingredients_test)
# greedy_model_test.to('cuda')
# greedy_test_loss_test, greedy_test_acc_test, greedy_test_f1_test, greedy_test_recall_test, greedy_test_kappa_test, greedy_test_auc_test = val_step(greedy_model_test, test_loader, train_loader, loss, 'cuda', 'MultiClass')
# print(greedy_test_loss_test, greedy_test_acc_test, greedy_test_f1_test, greedy_test_recall_test, greedy_test_kappa_test, greedy_test_auc_test)

# log_file.close()


final_val_df.to_csv(os.path.join(WEIGHT_PATH, 'FINAL_VAL_RESULTS.csv'))
final_test_df.to_csv(os.path.join(WEIGHT_PATH, 'FINAL_TEST_RESULTS.csv'))

import copy
print("Uniform souping ...")
state_dict_0 = torch.load(final_val_df.iloc[0]['model_path'])['model_state']
models_done = [1]
for index, row in final_val_df.iloc[1:].iterrows():
    print(row['model_path'])
    state_dict_1 = torch.load(row['model_path'])['model_state']
    alphal = [len(models_done)/(len(models_done)+1), 1- (len(models_done)/(len(models_done)+1))]
    models_done.append(index)
    model = get_model(MODEL, num_classes=NUM_CLASSES)
    uniform_model = souping(model, [state_dict_0, state_dict_1], alphal)
    uniform_model.to('cuda')
    # uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc = val_step(uniform_model, val_loader, train_loader, loss, 'cuda', 'MultiClass')
    # print(uniform_test_acc)
    state_dict_0 = copy.deepcopy(uniform_model.state_dict())
    print(index, models_done, len(models_done), alphal)
    # uniform_model.to('cuda')

uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc = val_step(uniform_model, test_loader, train_loader, loss, 'cuda', 'MultiClass')
print(uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc)
#greedy


import copy

print("Greedy souping ...")
model = get_model(MODEL, num_classes=NUM_CLASSES)

def greedy_souping_v2(NUM_CLASSES, val_loader, train_loader, loss, DEVICE, CLASSIFICATION, sort_by):
    # ranked_candidates = [i for i in range(len(val_models))]
    # ranked_candidates.sort(key=lambda x: -val_results[x])

        
    # ranked_candidates_names = [i for i in range(len(val_models))]
    # ranked_candidates_names.sort(key=lambda x: -val_results[x])

    current_best = final_val_df.iloc[0][val_sort_by]

    # print(ranked_candidates_names)

    print('currentttttttt bestttttttt', current_best, final_val_df.iloc[0]['model_path'])

    
    best_ingredients = [1]
    models_done = [1]
    state_dict_0 = torch.load(final_val_df.iloc[0]['model_path'])['model_state']
    # best_ingredients_names = ranked_candidates_names[:1]
    for index, row in final_val_df.iloc[1:].iterrows():
        print(row['model_path'])
        state_dict_1 = torch.load(row['model_path'])['model_state']
        alphal = [len(best_ingredients)/(len(best_ingredients)+1), 1- (len(best_ingredients)/(len(best_ingredients)+1))]
        model = get_model(MODEL, num_classes=NUM_CLASSES)
        uniform_model = souping(model, [state_dict_0, state_dict_1], alphal)
        uniform_model.to('cuda')


        # add current index to the ingredients
        ingredient_indices = best_ingredients + [index]
        # alphal = [0 for i in range(len(state_dicts))]
        # for j in ingredient_indices:
        #     alphal[j] = 1 / len(ingredient_indices)

        # benchmark and conditionally append
        greedy_val_loss, greedy_val_acc, greedy_val_f1, greedy_val_recall, greedy_val_kappa, greedy_val_auc = val_step(uniform_model, val_loader, train_loader, loss, DEVICE, CLASSIFICATION)
        if 'F1' in sort_by:            
            print(f'Models {ingredient_indices} got {greedy_val_f1} on validation.')
            if greedy_val_f1 > current_best:
                current_best = greedy_val_f1
                best_ingredients = ingredient_indices
                models_done.append(index)
                # best_ingredients_names = ingredient_names
                state_dict_0 = copy.deepcopy(uniform_model.state_dict())
                print(index, models_done, len(models_done), alphal)
        elif 'Recall' in sort_by:
            print(f'Models {ingredient_indices} got {greedy_val_recall} on validation.')
            if greedy_val_recall > current_best:
                current_best = greedy_val_recall
                best_ingredients = ingredient_indices
                models_done.append(index)

                # best_ingredients_names = ingredient_names
                state_dict_0 = copy.deepcopy(uniform_model.state_dict())
                print(index, models_done, len(models_done), alphal)
        elif 'Accuracy' in sort_by:
            print(f'Models {ingredient_indices} got {greedy_val_acc} on validation.')
            if greedy_val_acc > current_best:
                current_best = greedy_val_acc
                best_ingredients = ingredient_indices
                models_done.append(index)

                # best_ingredients_names = ingredient_names
                state_dict_0 = copy.deepcopy(uniform_model.state_dict())
                print(index, models_done, len(models_done), alphal)
        elif 'Kappa' in sort_by:
            print(f'Models {ingredient_indices} got {greedy_val_kappa} on validation.')
            if greedy_val_kappa > current_best:
                current_best = greedy_val_kappa
                best_ingredients = ingredient_indices
                models_done.append(index)

                # best_ingredients_names = ingredient_names
                state_dict_0 = copy.deepcopy(uniform_model.state_dict())
                print(index, models_done, len(models_done), alphal)
        elif 'AUC' in sort_by:
            print(f'Models {ingredient_indices} got {greedy_val_auc}% on validation.')
            if greedy_val_auc > current_best:
                current_best = greedy_val_auc
                best_ingredients = ingredient_indices
                models_done.append(index)
                # best_ingredients_names = ingredient_names
                state_dict_0 = copy.deepcopy(uniform_model.state_dict())
                print(index, models_done, len(models_done), alphal)
        print(best_ingredients)
    # alphal = [0 for i in range(len(state_dicts))]
    # for j in best_ingredients:
    #     alphal[j] = 1 / len(best_ingredients)
    # greedy_model = souping(model, state_dicts, alphal)

    return best_ingredients
# List of desired model names
best_ingredients = greedy_souping_v2(NUM_CLASSES, val_loader,train_loader, loss, 'cuda' , 'MultiClass', val_sort_by)
# greedy_model.to('cuda')
print('VAL INGREDIENTS',best_ingredients)

desired_models = best_ingredients

desired_models = [model-1 for model in desired_models]
print(desired_models)
# Models [1, 2, 3, 16, 17, 27, 35, 49] got 0.9822265625 on validation.

# Models [1, 2, 3, 16, 17, 27] got 0.9818359375 on validation.


# Filter dataframe based on desired model names and extract model paths
model_paths = final_val_df.iloc[desired_models]['model_path'].tolist()
print(model_paths)
# 0.850390625
state_dicts = []
for model in model_paths:
    state_dicts.append(torch.load(model)['model_state'])

alphas = [1/len(state_dicts)]* len(state_dicts)
print(alphas)
model = get_model(MODEL, num_classes=NUM_CLASSES)
greedy_model = souping(model, state_dicts, alphas)
greedy_model.to('cuda')

greedy_val_loss, greedy_val_acc, greedy_val_f1, greedy_val_recall, greedy_val_kappa, greedy_val_auc = val_step(greedy_model, val_loader, train_loader, loss, 'cuda', 'MultiClass')
print(greedy_val_loss, greedy_val_acc, greedy_val_f1, greedy_val_recall, greedy_val_kappa, greedy_val_auc)
greedy_test_loss, greedy_test_acc, greedy_test_f1, greedy_test_recall, greedy_test_kappa, greedy_test_auc = val_step(greedy_model, test_loader, train_loader, loss, 'cuda', 'MultiClass')
print(greedy_test_loss, greedy_test_acc, greedy_test_f1, greedy_test_recall, greedy_test_kappa, greedy_test_auc)

# greedy_test_loss_test, greedy_test_acc_test, greedy_test_f1_test, greedy_test_recall_test, greedy_test_kappa_test, greedy_test_auc_test = val_step(greedy_model_test, test_loader, train_loader, loss, 'cuda', 'MultiClass')
# print(greedy_test_loss_test, greedy_test_acc_test, greedy_test_f1_test, greedy_test_recall_test, greedy_test_kappa_test, greedy_test_auc_test)


log_file.close()
