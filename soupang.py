


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
import copy


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
CLASSIFICATION = str(config["CLASSIFICATION"])
val_sort_by = str(config["val_sort_by"])
test_sort_by = str(config["test_sort_by"])
TASK = str(config["TASK"])
PATHS = config["PATH"]
MODEL = config["MODEL"]
LOSS = config["LOSS"]
val_sort_by = config['val_sort_by']
test_sort_by = config['test_sort_by']

DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else 'cpu')


df = pd.read_csv('/home/santosh.sanjeev/Projects/model-soups/noodles/test/APTOS/DeiT-S/VAL_RESULTS.csv')
initial_model_path = '/home/santosh.sanjeev/Projects/model-soups/noodles/runs/san_final_hyp_models/san-initial_models/aptos_final_hyp/deitS_imagenet/2024-01-10_02-40-01/best_checkpoint.pth'




print('>>>>>>>>>>>>>>>>>>>>>>>>>>', DATASET,'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
state_dicts = []
val_results = []
test_results = []

#if there is no test directory create it
if not os.path.exists("test"):
    os.makedirs("test")

#if there is no folder named after dataset inside test directory create it
if not os.path.exists(os.path.join("test", DATASET, MODEL)):
    os.makedirs(os.path.join("test", DATASET, MODEL))

test_save_path = os.path.join("test", DATASET, MODEL)
# # Open a text file for logging
log_file_path = os.path.join("test", DATASET, MODEL, "output_log_angle_based_v2.txt")
log_file = open(log_file_path, "w")

# Redirect the standard output to the log file
sys.stdout = log_file
print('outtttttttttttt')
train_loader, val_loader, test_loader = get_dataset(DATASET, PATHS, "Minimal", 
                                                        PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK)



def get_xy(point, origin, vector_x, vector_y):
    return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])

# model1 =  getattr(models, 'resnet18')(pretrained=True)
# model1.fc = nn.Linear(model1.fc.in_features, 10)
# model1 = model1.to('cuda')

# val_accuracy_list = []
val_accuracy_list = [0]*(len(os.listdir(WEIGHT_PATH))-1)
print(len(val_accuracy_list))

#for each folder inside WEIGHT_PATH
model = get_model(MODEL, num_classes=NUM_CLASSES)
for idx, folder in enumerate(sorted(os.listdir(WEIGHT_PATH))):
    print(idx)
    if folder.startswith('testresults'):
        continue
    model_path = os.path.join(WEIGHT_PATH, folder)
    # read config in train_summary.json
    train_summary = json.load(open(os.path.join(model_path, "train_summary.json"), 'r'))
    model_config = train_summary["config"]

    #load model
    checkpoint = torch.load(os.path.join(model_path, "best_checkpoint.pth"), map_location=DEVICE)
    state_dicts.append(checkpoint['model'])

    model.load_state_dict(checkpoint['model'])
    # model = model.to(DEVICE)
    #load loss function
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
    


    # test_loss, test_acc, test_f1, test_recall, test_kappa, test_auc = val_step(model, test_loader, train_loader, loss, device = DEVICE, classification = CLASSIFICATION)
    # val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(model, val_loader, train_loader, loss, device = DEVICE, classification = CLASSIFICATION)

    # val_results.append({'Model Name': idx,
    #                                 'Val Accuracy': val_acc,
    #                                 'Val F1': val_f1,
    #                                 'Val Recall': val_recall,
    #                                 'Val Kappa': val_kappa,
    #                                 'Val AUC': val_auc,
    #                                 'Augmentation': augmentation,
    #                                 'Learning Rate': learning_rate,
    #                                 'Mixup': mixup,
    #                                 'SEED': seed})
    # test_results.append({'Model Name': idx,
    #                                 'Test Accuracy': test_acc,
    #                                 'Test F1': test_f1,
    #                                 'Test Recall': test_recall,
    #                                 'Test Kappa': test_kappa,
    #                                 'Test AUC': test_auc,
    #                                 'Augmentation': augmentation,
    #                                 'Learning Rate': learning_rate,
    #                                 'Mixup': mixup,
    #                                 'SEED': seed})

    # val_accuracy_list.append(val_acc)
    # test_accuracy_list.append(test_acc)

# np.save('my_list.npy', state_dicts)

# # Load the list back
# state_dicts = np.load('my_list.npy', allow_pickle=True)

# Print the loaded list
# results_test_df = pd.DataFrame(test_results)
# df = pd.DataFrame(val_results)
        


def angle_between_vectors(vectors):
    """
    Calculate the angle between the second and third vectors with respect to the origin.

    Parameters:
    - vectors: numpy array, each row represents a vector

    Returns:
    - angle_deg: float, angle in degrees
    """
    # Extract the second and third vectors from the array
    vector1 = vectors[1]
    vector2 = vectors[2]

    # Calculate vectors from origin to the given points
    origin = np.zeros_like(vector1)
    vector1_from_origin = vector1 - origin
    vector2_from_origin = vector2 - origin

    # Calculate the dot product
    dot_product = np.dot(vector1_from_origin, vector2_from_origin)

    # Calculate the magnitudes
    magnitude_vector1 = np.linalg.norm(vector1_from_origin)
    magnitude_vector2 = np.linalg.norm(vector2_from_origin)

    # Calculate the cosine of the angle
    cosine_theta = dot_product / (magnitude_vector1 * magnitude_vector2)

    # Calculate the angle in radians
    theta_rad = np.arccos(np.clip(cosine_theta, -1.0, 1.0))

    # Convert angle to degrees
    angle_deg = np.degrees(theta_rad)

    return angle_deg

WEIGHT_PATH_v2 = '/l/users/santosh.sanjeev/model_soups/runs/san_final_hyp_models/san-finetune/cifar100_final_hyp/deit-s/2024-01-18_19-46-03/'




best_model_acc = df.iloc[0][val_sort_by]
best_model_id = df.iloc[0]['Model Name']


model_data = []



for index, row in df.iloc[1:].iterrows():
    model_id = df.iloc[index]['Model Name']
    print(model_id)
    model = get_model(MODEL, num_classes=NUM_CLASSES)
    
    model = model.to(DEVICE)
    model1 = get_model(MODEL, num_classes=NUM_CLASSES)
    model1.load_state_dict(torch.load(initial_model_path,map_location='cuda')['model'])
    model1 = model1.to(DEVICE)
    model2 = get_model(MODEL, num_classes=NUM_CLASSES)
    model3 = get_model(MODEL, num_classes=NUM_CLASSES)
    model2.load_state_dict(state_dicts[best_model_id])
    model3.load_state_dict(state_dicts[model_id])
    model2 = model2.to(DEVICE)
    model3 = model3.to(DEVICE)

    w = list()
    curve_parameters = [model1.parameters(), model2.parameters(), model3.parameters()]
    for i in range(3):
        w.append(np.concatenate([
            p.data.cpu().numpy().ravel() for p in curve_parameters[i]
        ]))

    u = w[2] - w[0]
    dx = np.linalg.norm(u)
    u /= dx
    v = w[1] - w[0]
    v -= np.dot(u, v) * u
    dy = np.linalg.norm(v)
    v /= dy
    bend_coordinates = np.stack(get_xy(p, w[0], u, v) for p in w)
    angle = angle_between_vectors(bend_coordinates)
    model_data.append((model_id, angle))

# Sort the model IDs based on angles in descending order
sorted_model_data = sorted(model_data, key=lambda x: x[1], reverse=True)

# Extract sorted model IDs and angles
sorted_model_ids = [model_id for model_id, angle in sorted_model_data]
sorted_angles = [angle for model_id, angle in sorted_model_data]

print("Sorted Model IDs based on angle:")
print(sorted_model_ids)
print("Corresponding Angles:")
print(sorted_angles)



#     # for i in range(1, len(state_dicts)):
#     #     # add current index to the ingredients
#     #     ingredient_indices = best_ingredients + [ranked_candidates[i]]
#     #     ingredient_names = best_ingredients_names + [ranked_candidates_names[i]]
#     #     alphal = [0 for i in range(len(state_dicts))]
#     #     for j in ingredient_indices:
#     #         alphal[j] = 1 / len(ingredient_indices)

#     #     # benchmark and conditionally append
#     #     model = get_model(model_config, num_classes=NUM_CLASSES)

#     #     greedy_model = souping(model, state_dicts, alphal)
#     #     greedy_model.to(DEVICE)
#     #     greedy_val_loss, greedy_val_acc, greedy_val_f1, greedy_val_recall, greedy_val_kappa, greedy_val_auc = val_step(greedy_model, val_loader, train_loader, loss, DEVICE, CLASSIFICATION)
#     #     if 'F1' in sort_by:            
#     #         print(f'Models {ingredient_names} got {greedy_val_f1} on validation.')
#     #         if greedy_val_f1 > current_best:
#     #             current_best = greedy_val_f1
#     #             best_ingredients = ingredient_indices
#     #             best_ingredients_names = ingredient_names
#     #     elif 'Recall' in sort_by:
#     #         print(f'Models {ingredient_names} got {greedy_val_recall} on validation.')
#     #         if greedy_val_recall > current_best:
#     #             current_best = greedy_val_recall
#     #             best_ingredients = ingredient_indices
#     #             best_ingredients_names = ingredient_names

#     #     elif 'Accuracy' in sort_by:
#     #         print(f'Models {ingredient_names} got {greedy_val_acc} on validation.')
#     #         if greedy_val_acc > current_best:
#     #             current_best = greedy_val_acc
#     #             best_ingredients = ingredient_indices
#     #             best_ingredients_names = ingredient_names

#     #     elif 'Kappa' in sort_by:
#     #         print(f'Models {ingredient_names} got {greedy_val_kappa} on validation.')
#     #         if greedy_val_kappa > current_best:
#     #             current_best = greedy_val_kappa
#     #             best_ingredients = ingredient_indices
#     #             best_ingredients_names = ingredient_names

#     #     elif 'AUC' in sort_by:
#     #         print(f'Models {ingredient_names} got {greedy_val_auc}% on validation.')
#     #         if greedy_val_auc > current_best:
#     #             current_best = greedy_val_auc
#     #             best_ingredients = ingredient_indices
#     #             best_ingredients_names = ingredient_names

#     # alphal = [0 for i in range(len(state_dicts))]
#     # for j in best_ingredients:
#     #     alphal[j] = 1 / len(best_ingredients)
#     # greedy_model = souping(model, state_dicts, alphal)

#     # return greedy_model, best_ingredients_names



# for index, row in df.iloc[1:].iterrows():
#     for j_index, j_row in df.iloc[index:].iterrows():
#         print(df.iloc[j_index]['Model Name'])
#         model_acc = df.iloc[j_index]['Val Accuracy']
#         model_id = df.iloc[j_index]['Model Name']
#         model = get_model(MODEL, num_classes=NUM_CLASSES)
#         model = model.to(DEVICE)
#         model1 = get_model(MODEL, num_classes=NUM_CLASSES)
#         model1 = model1.to(DEVICE)
#         model2 = get_model(MODEL, num_classes=NUM_CLASSES)
#         model3 = get_model(MODEL, num_classes=NUM_CLASSES)
#         model2.load_state_dict(state_dicts[best_model_acc])
#         model3.load_state_dict(state_dicts[model_id])
#         model2 = model2.to(DEVICE)
#         model3 = model3.to(DEVICE)

#         w = list()
#         curve_parameters = [model1.parameters(),model2.parameters(),model3.parameters()]#list(curve_model.net.parameters())
#         for i in range(3):
#             w.append(np.concatenate([
#                 p.data.cpu().numpy().ravel() for p in curve_parameters[i]#[i::args.num_bends]
#             ]))
#         u = w[2] - w[0]
#         dx = np.linalg.norm(u)
#         u /= dx
#         v = w[1] - w[0]
#         v -= np.dot(u, v) * u
#         dy = np.linalg.norm(v)
#         v /= dy
#         bend_coordinates = np.stack(get_xy(p, w[0], u, v) for p in w)
#         angle = angle_between_vectors(bend_coordinates)
#         alphal = [0.5,0.5]
#         uniform_model = souping(model, [state_dicts[combination_sets[0]], state_dicts[combination_sets[1]]], alphal)
#         uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc = val_step(uniform_model, test_loader, train_loader, loss, DEVICE, CLASSIFICATION)

#         print(angle, uniform_test_acc)

#     points = uniform_test_acc - 0.5*(val_accuracy_list[combination_sets[0]] + val_accuracy_list[combination_sets[1]])






best_model_acc = df.iloc[0][val_sort_by]
best_model_id = df.iloc[0]['Model Name']
best_ls = [best_model_id]
model_data = []
ls = sorted_model_ids#[33, 30, 34, 37, 32, 31, 38, 41, 36, 35, 39, 21, 18, 22, 25, 20, 19, 26, 29, 23, 24, 40, 0, 3, 27, 12, 42, 45, 14, 13, 16, 15, 17, 7, 2, 4, 1, 44, 49, 46, 43, 50, 53, 47, 48, 8, 11, 51, 5, 6, 52, 9, 10]
dkfjsdfk = 0
while True:
    if dkfjsdfk >= len(ls):
        break
    model_id = ls[dkfjsdfk]
    print(model_id)
    model = get_model(MODEL, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    model1 = get_model(MODEL, num_classes=NUM_CLASSES)
    model1.load_state_dict(torch.load(initial_model_path,map_location='cuda')['model'])
    model1 = model1.to(DEVICE)
    model2 = get_model(MODEL, num_classes=NUM_CLASSES)
    model3 = get_model(MODEL, num_classes=NUM_CLASSES)
    model2.load_state_dict(state_dicts[best_model_id])
    model3.load_state_dict(state_dicts[model_id])
    model2 = model2.to(DEVICE)
    model3 = model3.to(DEVICE)

    sd = [state_dicts[best_model_id], state_dicts[model_id]]
    alphal = [len(best_ls)/(len(best_ls)+1),1- (len(best_ls)/(len(best_ls)+1))]
    # alphal = [0.5,0.5]
    uniform_model = souping(model, sd, alphal)
    uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc = val_step(uniform_model, val_loader, train_loader, loss, DEVICE, CLASSIFICATION)
    print(uniform_test_f1, best_model_acc)
    if uniform_test_f1 > best_model_acc:
        print('in uniform testtttttttttttttttt')
        best_model_acc = uniform_test_f1
        state_dicts[best_model_id] = uniform_model.state_dict()
        best_ls.append(model_id)
        print(uniform_test_f1, best_ls)
        ls.remove(model_id)
        print(ls)#ls[ls.index(model_id) + 1:]
        model2 = get_model(MODEL, num_classes=NUM_CLASSES)
        model3 = get_model(MODEL, num_classes=NUM_CLASSES)

        model2.load_state_dict(state_dicts[best_model_id])
        model2 = model2.to(DEVICE)
        angles_with_remaining_models = []
        for j in ls:
            model_j_id = j
            model3 = get_model(MODEL, num_classes=NUM_CLASSES)
            model3.load_state_dict(state_dicts[model_j_id])
            model3 = model3.to(DEVICE)


            w = list()
            curve_parameters = [model1.parameters(), model2.parameters(), model3.parameters()]
            for i in range(3):
                w.append(np.concatenate([
                    p.data.cpu().numpy().ravel() for p in curve_parameters[i]
                ]))

            u = w[2] - w[0]
            dx = np.linalg.norm(u)
            u /= dx
            v = w[1] - w[0]
            v -= np.dot(u, v) * u
            dy = np.linalg.norm(v)
            v /= dy
            bend_coordinates = np.stack(get_xy(p, w[0], u, v) for p in w)
            angle = angle_between_vectors(bend_coordinates)
            angles_with_remaining_models.append((model_j_id, angle))

        # Sort ls based on angles in descending order
        ls = [model_id for model_id, _ in sorted(angles_with_remaining_models, key=lambda x: x[1], reverse=True)]

        # Print the sorted ls and corresponding angles
        print("Sorted Model IDs based on angle:")
        print(ls)
        print("Corresponding Angles:")
        print([angle for _, angle in sorted(angles_with_remaining_models, key=lambda x: x[1], reverse=True)])
        dkfjsdfk = 0
    else:
        dkfjsdfk+=1


model = get_model(MODEL, num_classes=NUM_CLASSES)
model = model.to(DEVICE)
sd = []
for i in best_ls:
    sd.append(state_dicts[i])
alphal = [1/len(best_ls)]*len(best_ls)
# alphal = [0.25,0.25, 0.25,0.25]
uniform_model = souping(model, sd, alphal)
uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc = val_step(uniform_model, val_loader, train_loader, loss, DEVICE, CLASSIFICATION)
print(uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc)

uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc = val_step(uniform_model, test_loader, train_loader, loss, DEVICE, CLASSIFICATION)

print(uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc)
log_file.close()