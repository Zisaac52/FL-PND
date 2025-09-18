"""
fl-pnd: A Flower / PyTorch app for Semantic Segmentation.
This file defines the model, training, and testing logic.
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import warnings
from torch.optim.lr_scheduler import StepLR

def get_net():
    NUM_CLASSES = 5
    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1)
    return model

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, trainloader, epochs: int, device: torch.device, class_weights: torch.Tensor):
    """Trains the model on a given dataset, ignoring the background class."""
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=0)
    
    for param in net.backbone.parameters():
        param.requires_grad = False
        
    params_to_train = net.classifier.parameters()
    optimizer = torch.optim.Adam(params_to_train, lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, masks in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(trainloader)
        print(f"Epoch {epoch+1} training loss (foreground only): {avg_epoch_loss:.4f}")
        scheduler.step()

    for param in net.backbone.parameters():
        param.requires_grad = True

def _update_confusion_matrix(gt_label, pred_label, confusion_matrix):
    gt_flat = gt_label.flatten()
    pred_flat = pred_label.flatten()
    mask = (gt_flat >= 0) & (gt_flat < confusion_matrix.shape[0])
    conf_index = confusion_matrix.shape[0] * gt_flat[mask].astype(int) + pred_flat[mask]
    hist = np.bincount(conf_index, minlength=confusion_matrix.size)
    confusion_matrix += hist.reshape(confusion_matrix.shape)

def _get_metrics_from_confusion_matrix(confusion_matrix):
    """Calculates mIoU from the confusion matrix, ignoring the background class."""
    fg_confusion_matrix = confusion_matrix[1:, 1:]
    
    intersection = np.diag(fg_confusion_matrix)
    union = np.sum(fg_confusion_matrix, axis=1) + np.sum(fg_confusion_matrix, axis=0) - intersection
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        iou = intersection / union

    mean_iou = np.nanmean(iou)
    return mean_iou

def test(net, testloader, device: torch.device):
    """Evaluates the model on a given dataset, calculating loss and mIoU for foreground classes."""
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    
    num_classes = net.classifier[4].out_channels
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    net.eval()
    with torch.no_grad():
        for images, masks in tqdm(testloader, desc="Evaluating"):
            images = images.to(device)
            masks_cpu = masks.cpu().numpy().astype(np.int32)
            masks = masks.to(device)
            
            outputs = net(images)['out']
            loss = criterion(outputs, masks)
            if not torch.isnan(loss):
                total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.int32)
            _update_confusion_matrix(masks_cpu, preds, confusion_matrix)

    avg_loss = total_loss / len(testloader)
    mean_iou = _get_metrics_from_confusion_matrix(confusion_matrix)
    
    return avg_loss, mean_iou