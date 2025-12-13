import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import torchvision.transforms.functional as TF

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        f_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean': return torch.mean(f_loss)
        elif self.reduction == 'sum': return torch.sum(f_loss)
        return f_loss

def mixup_data(x, y, alpha=0.4, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    index = torch.randperm(x.size(0)).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device):
        self.model = model
        self.opt = optimizer
        self.sched = scheduler
        self.crit = criterion
        self.device = device
        self.scaler = torch.amp.GradScaler('cuda')

    def train_epoch(self, loader):
        self.model.train()
        avg_loss = 0
        
        for x, y in tqdm(loader, desc="Train", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4, device=self.device)
            
            self.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = self.model(x)
                loss = mixup_criterion(self.crit, out, y_a, y_b, lam)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            avg_loss += loss.item()
        return avg_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        avg_loss = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            loss = self.crit(out, y)
            avg_loss += loss.item()
        return avg_loss / len(loader)

@torch.no_grad()
def run_tta(model, loader, device):
    model.eval()
    all_preds = []
    for images, _ in tqdm(loader, desc="TTA Inference", leave=False):
        images = images.to(device)
        p1 = model(images).softmax(1)
        p2 = model(TF.hflip(images)).softmax(1)
        p3 = model(TF.vflip(images)).softmax(1)
        p_avg = (p1 + p2 + p3) / 3.0
        all_preds.append(p_avg.cpu().numpy())
    return np.vstack(all_preds)