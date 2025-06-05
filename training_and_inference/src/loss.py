import torch
import numpy as np
import torch.nn.functional as F

def MSE_loss(y_pred, target):
    y_pred = y_pred.squeeze()
    target = target.squeeze()
    loss = torch.mean((y_pred - target) ** 2)
    return loss

# MAE Loss
def MAE_loss(y_pred, target):
    y_pred = y_pred.squeeze()
    target = target.squeeze()
    
    loss = torch.mean(torch.abs(y_pred - target))
    return loss

# Huber Loss
def Huber_loss(y_pred, target, delta=0.1e6):
    y_pred = y_pred.squeeze()
    target = target.squeeze()
    
    loss = F.huber_loss(y_pred, target, delta=delta, reduction='mean')
    return loss

