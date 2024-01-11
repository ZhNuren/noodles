import argparse
import numpy as np
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import utils
import clip
# from engine import trainer, val_step
# from utils import plot_results
# from models import get_model

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
import train_utils
from utils import EarlyStopper
from torch.utils.data.dataset import Subset



parser = argparse.ArgumentParser(description='FGE training')

parser.add_argument('--dir', type=str, default='/tmp/fge/', metavar='DIR',
                    help='training directory (default: /tmp/fge)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--cycle', type=int, default=4, metavar='N',
                    help='number of epochs to train (default: 4)')
parser.add_argument('--lr_1', type=float, default=0.05, metavar='LR1',
                    help='initial learning rate (default: 0.05)')
parser.add_argument('--lr_2', type=float, default=0.0001, metavar='LR2',
                    help='initial learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

assert args.cycle % 2 == 0, 'Cycle length should be even'

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'fge.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)

torch.cuda.manual_seed(args.seed)

CIFAR_PATH = "./dataset/cifar"
CIFAR_INDICES = "./dataset/cifar/cifar1098_idxs.npy"

train_transform = v2.Compose([
    v2.RandomResizedCrop((224, 224), scale=(0.7, 1.2), antialias=True),
    # v2.RandomRotation(degrees=(-70, 70)),
    # v2.RandomAffine(degrees=(-15, 15), translate=(0.25, 0.25), scale=(0.7, 1.2), shear=(-15, 15, -15, 15)),
    # v2.RandomPerspective(distortion_scale=0.2, p=0.2),
    # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    # v2.RandomAutocontrast(p=0.2),
    # v2.RandomEqualize(p=0.2),
    v2.RandAugment(num_ops=2, magnitude=15, interpolation = InterpolationMode.BILINEAR),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
])

val_transform = v2.Compose([
    v2.Resize((224, 224), antialias=True),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])


trainset = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=True, transform=train_transform)
valset = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=True, transform=val_transform)
test_dataset = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=False, transform=val_transform)

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
print(len(val_dataset), len(train_dataset))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=12)

template = train_utils.openai_imagenet_template



pre_model, _ = clip.load('ViT-B/32')
# checkpoint1 = torch.load('/l/users/santosh.sanjeev/model_soups/final_models/CLIP_wo_mixup/2024-01-07_02-48-23/hparam_2024-01-07_06-45-10/best_checkpoint.pth')
checkpoint1 = torch.load('/l/users/santosh.sanjeev/model_soups/final_models/CLIP_wo_mixup/2024-01-07_02-48-23/hparam_2024-01-07_03-47-20/best_checkpoint.pth')

criterion = torch.nn.CrossEntropyLoss()

feature_dim = checkpoint1['model']['classification_head.weight'].shape[1]
num_classes = checkpoint1['model']['classification_head.weight'].shape[0]
normalize = True
clf = train_utils.zeroshot_classifier(pre_model, test_dataset.classes, template, 'cuda')
model = train_utils.ModelWrapper(pre_model, feature_dim, num_classes, normalize, clf)
model.load_state_dict(checkpoint1['model'])
for pp in model.parameters():
    pp.data = pp.data.float()
model = model.to('cuda')


start_epoch = 10 #checkpoint['hyperparameters']['epochs'] + 1
# model.load_state_dict(checkpoint['model_state'])
model.cuda()

test_res = train_utils.test(val_loader, model, criterion)
print('111111111',test_res['accuracy'])

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr_1,
)
optimizer.load_state_dict(checkpoint1['optimizer'])

ensemble_size = 0
predictions_sum = np.zeros((len(val_loader.dataset), num_classes))

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'ens_acc', 'time']

for epoch in range(args.epochs):
    time_ep = time.time()
    lr_schedule = train_utils.cyclic_learning_rate(epoch, args.cycle, args.lr_1, args.lr_2)
    # print(lr_schedule)
    train_res = train_utils.train(train_loader, model, optimizer, criterion, lr_schedule=lr_schedule)
    test_res = train_utils.test(val_loader, model, criterion)
    time_ep = time.time() - time_ep
    predictions, targets = train_utils.predictions(val_loader, model)
    ens_acc = None
    if (epoch % args.cycle + 1) == args.cycle // 2:
        ensemble_size += 1
        predictions_sum += predictions
        ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)

    if (epoch + 1) % (args.cycle // 2) == 0:
        print(epoch)
        train_utils.save_checkpoint(
            args.dir,
            start_epoch + epoch,
            name='fge',
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

    values = [epoch, lr_schedule(1.0), train_res['loss'], train_res['accuracy'], test_res['nll'],
              test_res['accuracy'], ens_acc, time_ep]
    # table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.8f')
    testttttt = train_utils.test(test_loader, model, criterion)
    print(testttttt['accuracy'])
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)





