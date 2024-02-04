import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import os

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


# Assuming `state_dicts_list` is your list of model state dicts
# Assuming `model` is an instance of your model class
WEIGHT_PATH_v2 = '/l/users/santosh.sanjeev/model_soups/runs/san_final_hyp_models/san-finetune/aptos_final_hyp/deitS_imagenet/2024-01-10_02-53-56/'
initial_val_df = pd.read_csv('/home/santosh.sanjeev/Projects/model-soups/noodles/test/APTOS/DeiT-S/VAL_RESULTS.csv')
initial_test_df = pd.read_csv('/home/santosh.sanjeev/Projects/model-soups/noodles/test/APTOS/DeiT-S/TEST_RESULTS.csv')
NUM_CLASSES = 5
MODEL = 'DeiT-S'
DEVICE = 'cuda'
val_sort_by = 'Val F1'
# initial_model_path = '/home/santosh.sanjeev/Projects/model-soups/noodles/runs/san_final_hyp_models/san-initial_models/aptos_final_hyp/deitS_imagenet/2024-01-10_02-40-01/best_checkpoint.pth'


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

# Function to convert model parameters to vectors
def parameters_to_vector(parameters):
    return np.concatenate([param.detach().cpu().numpy().flatten() for param in parameters])
# Convert state dicts to vectors
vectors = [parameters_to_vector(state_dict.values()) for state_dict in state_dicts]
vectors = np.array(vectors)

num_clusters = 3  # You can change this number as needed

# K-Means Clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(vectors)
cluster_labels = kmeans.labels_

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

plt.figure(figsize=(8, 6))
for cluster_label in range(num_clusters):
    plt.scatter(reduced_vectors[cluster_labels == cluster_label, 0],
                reduced_vectors[cluster_labels == cluster_label, 1],
                label=f'Cluster {cluster_label}')

plt.title('K-Means Clustering of Model State Dicts')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()
