import torch

from matplotlib import pyplot as plt
import numpy as np
import sklearn
from scipy.interpolate import interp1d
from scipy.integrate import quad
import cv2
from sklearn.metrics import confusion_matrix


def plot_results(results, save_dir, name = None):

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(results['train_loss'], label='Train loss')
    plt.plot(results['val_loss'], label='Validation loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(results['train_acc'], label='Train accuracy')
    plt.plot(results['val_acc'], label='Validation accuracy')
    plt.legend()
    if name:
        plt.savefig(save_dir + name)
    else:
        plt.savefig(save_dir + '/LossAccuracy.png')

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
