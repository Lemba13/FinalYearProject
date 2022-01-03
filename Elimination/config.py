from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import torch


BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
STEP_SIZE = 10
GAMMA = 0.5
EPOCH_THRES = 8
WEIGHT_DECAY = 0.00001
ENCODED_DIM = 32
PATH = 'weights/baseline_model_steganalysis.pt'
PATH0 = 'weights/baseline_model_elimination.pt'

transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_tag, dim=1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc
