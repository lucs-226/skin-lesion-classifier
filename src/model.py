import torch.nn as nn
from torchvision import models
from .dataset import WEIGHTS

def build_model(num_classes, device):
    model = models.efficientnet_b3(weights=WEIGHTS)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model.to(device)