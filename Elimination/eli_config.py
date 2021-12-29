from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import torch

transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                       ToTensorV2(),
                       ])
