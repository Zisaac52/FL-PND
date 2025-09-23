# """
# fl-pnd: A Flower / PyTorch app for Semantic Segmentation.
# This file defines the model, training, and testing logic.
# """
# from collections import OrderedDict
# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision
# from tqdm import tqdm
# import warnings
# from torch.optim.lr_scheduler import StepLR

# def get_net():
#     NUM_CLASSES = 5
#     weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
#     model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
#     in_channels = model.classifier[4].in_channels
#     model.classifier[4] = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1)
#     return model

# def set_parameters(net, parameters):
#     params_dict = zip(net.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#     net.load_state_dict(state_dict, strict=True)

# def train(net, trainloader, epochs: int, device: torch.device, class_weights: torch.Tensor):
#     """Trains the model on a given dataset, ignoring the background class."""
#     criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=0)
    
#     for param in net.backbone.parameters():
#         param.requires_grad = False
        
#     params_to_train = net.classifier.parameters()
#     optimizer = torch.optim.Adam(params_to_train, lr=1e-3) #1e-4 和 1e-3 都可以 
#     scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

#     net.train()
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         for images, masks in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
#             images, masks = images.to(device), masks.to(device)
            
#             optimizer.zero_grad()
#             outputs = net(images)['out']
#             loss = criterion(outputs, masks)
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()
        
#         avg_epoch_loss = epoch_loss / len(trainloader)
#         print(f"Epoch {epoch+1} training loss (foreground only): {avg_epoch_loss:.4f}")
#         scheduler.step()

#     for param in net.backbone.parameters():
#         param.requires_grad = True

# def _update_confusion_matrix(gt_label, pred_label, confusion_matrix):
#     gt_flat = gt_label.flatten()
#     pred_flat = pred_label.flatten()
#     mask = (gt_flat >= 0) & (gt_flat < confusion_matrix.shape[0])
#     conf_index = confusion_matrix.shape[0] * gt_flat[mask].astype(int) + pred_flat[mask]
#     hist = np.bincount(conf_index, minlength=confusion_matrix.size)
#     confusion_matrix += hist.reshape(confusion_matrix.shape)

# def _get_metrics_from_confusion_matrix(confusion_matrix):
#     """
#     Calculates multiple metrics from the confusion matrix.
#     - Foreground Pixel Accuracy (ignores background)
#     - Foreground mIoU (ignores background)
#     """
#     # [核心修改] 我们只关注前景类别 (从索引1开始)
#     fg_confusion_matrix = confusion_matrix[1:, 1:]
    
#     # 1. 计算前景像素准确率 (Foreground Pixel Accuracy)
#     #    (前景对角线元素之和) / (所有前景元素之和)
#     fg_correct = np.sum(np.diag(fg_confusion_matrix))
#     fg_total = np.sum(fg_confusion_matrix)
#     fg_pixel_accuracy = fg_correct / (fg_total + 1e-15)

#     # 2. 计算前景 mIoU (Foreground mIoU)
#     intersection = np.diag(fg_confusion_matrix)
#     union = np.sum(fg_confusion_matrix, axis=1) + np.sum(fg_confusion_matrix, axis=0) - intersection
    
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", RuntimeWarning)
#         iou = intersection / union

#     mean_iou = np.nanmean(iou)
    
#     # 3. 返回一个包含所有前景指标的字典
#     metrics_dict = {
#         "fg_pixel_accuracy": fg_pixel_accuracy,
#         "mean_iou": mean_iou
#     }
    
#     print(f"\n--- Calculated Metrics (Foreground Only) ---")
#     print(f"  Foreground Pixel Accuracy: {fg_pixel_accuracy:.4f}")
#     print(f"  Mean IoU:                  {mean_iou:.4f}")
#     print(f"------------------------------------------\n")

#     return metrics_dict

# def test(net, testloader, device: torch.device):
#     """
#     Evaluates the model and returns loss and a dictionary of metrics.
#     """
#     criterion = nn.CrossEntropyLoss(ignore_index=0)
#     total_loss = 0.0
    
#     num_classes = net.classifier[4].out_channels
#     confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

#     net.eval()
#     with torch.no_grad():
#         for images, masks in tqdm(testloader, desc="Evaluating"):
#             # ... (forward pass and loss calculation remain the same) ...
#             images = images.to(device)
#             masks_cpu = masks.cpu().numpy().astype(np.int32)
#             masks = masks.to(device)
            
#             outputs = net(images)['out']
#             loss = criterion(outputs, masks)
#             if not torch.isnan(loss):
#                 total_loss += loss.item()

#             preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.int32)
#             _update_confusion_matrix(masks_cpu, preds, confusion_matrix)

#     avg_loss = total_loss / len(testloader)
    
#     # [核心修改] _get_metrics... 现在返回一个字典
#     metrics_dict = _get_metrics_from_confusion_matrix(confusion_matrix)
    
#     # 返回 loss 和 metrics 字典
#     return avg_loss, metrics_dict
#     """Evaluates the model on a given dataset, calculating loss and mIoU for foreground classes."""
#     criterion = nn.CrossEntropyLoss(ignore_index=0)
#     total_loss = 0.0
    
#     num_classes = net.classifier[4].out_channels
#     confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

#     net.eval()
#     with torch.no_grad():
#         for images, masks in tqdm(testloader, desc="Evaluating"):
#             images = images.to(device)
#             masks_cpu = masks.cpu().numpy().astype(np.int32)
#             masks = masks.to(device)
            
#             outputs = net(images)['out']
#             loss = criterion(outputs, masks)
#             if not torch.isnan(loss):
#                 total_loss += loss.item()

#             preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.int32)
#             _update_confusion_matrix(masks_cpu, preds, confusion_matrix)

#     avg_loss = total_loss / len(testloader)
#     mean_iou = _get_metrics_from_confusion_matrix(confusion_matrix)
    
#     return avg_loss, mean_iou


# """
# fl-pnd: A Flower / PyTorch app for Semantic Segmentation.
# This file defines the model (a pre-trained DeepLabV3+), training, and testing logic.
# """
# from collections import OrderedDict
# import numpy as np
# import torch  # <--- [核心修复] 添加这一行
# import torch.nn as nn
# import torchvision
# from tqdm import tqdm
# import warnings
# from torch.optim.lr_scheduler import StepLR

# # ==============================================================================
# # 1. 定义模型获取函数 (使用预训练的 DeepLabV3+)
# # ==============================================================================
# def get_net():
#     """Instantiates and returns a pre-trained DeepLabV3+ model with a modified classifier."""
#     NUM_CLASSES = 5
#     weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
#     model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)

#     in_channels = model.classifier[4].in_channels
#     model.classifier[4] = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1)
    
#     return model

# # ==============================================================================
# # 2. 定义模型参数的设置函数
# # ==============================================================================
# def set_parameters(net, parameters):
#     """Updates a local model with parameters received from the server."""
#     params_dict = zip(net.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#     net.load_state_dict(state_dict, strict=True)

# # ==============================================================================
# # 3. 定义训练和测试函数
# # ==============================================================================
# def train(
#     net, 
#     trainloader, 
#     epochs: int, 
#     device: torch.device, 
#     class_weights: torch.Tensor,
#     current_round: int
# ):
#     """Trains the model with a two-stage fine-tuning strategy."""
#     criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=0)
    
#     FREEZE_UNTIL_ROUND = 30 

#     if current_round <= FREEZE_UNTIL_ROUND:
#         print(f"--- [Stage 1] Fine-tuning classifier head (lr=1e-3) ---")
#         for param in net.backbone.parameters():
#             param.requires_grad = False
        
#         for param in net.classifier.parameters():
#             param.requires_grad = True

#         params_to_train = net.classifier.parameters()
#         optimizer = torch.optim.Adam(params_to_train, lr=1e-3)
#         scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
#     else:
#         print(f"--- [Stage 2] End-to-end fine-tuning (lr=1e-5) ---")
#         for param in net.parameters():
#             param.requires_grad = True
            
#         optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
#         scheduler = StepLR(optimizer, step_size=100, gamma=1.0)

#     net.train()
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         for images, masks in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
#             images, masks = images.to(device), masks.to(device)
#             optimizer.zero_grad()
#             outputs = net(images)['out']
#             loss = criterion(outputs, masks)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
        
#         avg_epoch_loss = epoch_loss / len(trainloader)
#         print(f"Epoch {epoch+1} training loss (foreground only): {avg_epoch_loss:.4f}")
#         scheduler.step()

# # --- mIoU Helper Functions ---
# def _update_confusion_matrix(gt_label, pred_label, confusion_matrix):
#     gt_flat = gt_label.flatten()
#     pred_flat = pred_label.flatten()
#     mask = (gt_flat >= 0) & (gt_flat < confusion_matrix.shape[0])
#     conf_index = confusion_matrix.shape[0] * gt_flat[mask].astype(int) + pred_flat[mask]
#     hist = np.bincount(conf_index, minlength=confusion_matrix.size)
#     confusion_matrix += hist.reshape(confusion_matrix.shape)

# def _get_metrics_from_confusion_matrix(confusion_matrix):
#     """Calculates multiple metrics from the confusion matrix."""
#     fg_confusion_matrix = confusion_matrix[1:, 1:]
    
#     fg_correct = np.sum(np.diag(fg_confusion_matrix))
#     fg_total = np.sum(fg_confusion_matrix)
#     fg_pixel_accuracy = fg_correct / (fg_total + 1e-15)

#     intersection = np.diag(fg_confusion_matrix)
#     union = np.sum(fg_confusion_matrix, axis=1) + np.sum(fg_confusion_matrix, axis=0) - intersection
    
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", RuntimeWarning)
#         iou = intersection / union

#     mean_iou = np.nanmean(iou)
    
#     metrics_dict = {
#         "fg_pixel_accuracy": fg_pixel_accuracy,
#         "mean_iou": mean_iou
#     }
    
#     return metrics_dict

# def test(net, testloader, device: torch.device):
#     """Evaluates the model and returns loss and a dictionary of metrics."""
#     criterion = nn.CrossEntropyLoss(ignore_index=0)
#     total_loss = 0.0
    
#     num_classes = net.classifier[4].out_channels
#     confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

#     net.eval()
#     with torch.no_grad():
#         for images, masks in tqdm(testloader, desc="Evaluating"):
#             images = images.to(device)
#             masks_cpu = masks.cpu().numpy().astype(np.int32)
#             masks = masks.to(device)
            
#             outputs = net(images)['out']
#             loss = criterion(outputs, masks)
#             if not torch.isnan(loss):
#                 total_loss += loss.item()

#             preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.int32)
#             _update_confusion_matrix(masks_cpu, preds, confusion_matrix)

#     avg_loss = total_loss / len(testloader)
#     metrics_dict = _get_metrics_from_confusion_matrix(confusion_matrix)
    
#     return avg_loss, metrics_dict


"""
fl-pnd: A Flower / PyTorch app for Semantic Segmentation.
This file defines the model, training, and testing logic.

Current Version:
- Model: U-Net with a pre-trained ResNet-50 backbone.
- Loss Function: Weighted Cross-Entropy Loss (ignoring background).
- Training Strategy: Two-stage fine-tuning.
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import warnings
from torch.optim.lr_scheduler import StepLR

# --- 导入 U-Net 模型库 ---
import segmentation_models_pytorch as smp

# ==============================================================================
# 1. 定义模型获取函数 (U-Net)
# ==============================================================================
def get_net():
    """Instantiates a U-Net model with a pre-trained ResNet-50 backbone."""
    NUM_CLASSES = 5
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model

# ==============================================================================
# 2. 定义模型参数的设置函数
# ==============================================================================
def set_parameters(net, parameters):
    """Updates a local model with parameters received from the server."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# ==============================================================================
# 3. 定义训练和测试函数
# ==============================================================================
def train(
    net, 
    trainloader, 
    epochs: int, 
    device: torch.device, 
    class_weights: torch.Tensor,
    current_round: int
):
    """Trains the model using Weighted Cross-Entropy Loss and a two-stage strategy."""
    
    # --- [核心修改] 恢复使用稳定、可靠的 CrossEntropyLoss ---
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=0)
    
    # --- 两阶段微调逻辑 (适配 U-Net) ---
    FREEZE_UNTIL_ROUND = 30 
    if current_round <= FREEZE_UNTIL_ROUND:
        print(f"--- [Stage 1] U-Net: Fine-tuning decoder (lr=1e-3) ---")
        for param in net.encoder.parameters():
            param.requires_grad = False
        for param in net.decoder.parameters():
            param.requires_grad = True
        params_to_train = net.decoder.parameters()
        optimizer = torch.optim.Adam(params_to_train, lr=1e-3)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        print(f"--- [Stage 2] U-Net: End-to-end fine-tuning (lr=1e-5) ---")
        for param in net.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
        scheduler = StepLR(optimizer, step_size=100, gamma=1.0)

    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, masks in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            # SMP 模型直接返回张量
            outputs = net(images)
            
            # 计算损失
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(trainloader)
        print(f"Epoch {epoch+1} training loss (foreground only): {avg_epoch_loss:.4f}")
        scheduler.step()

# --- mIoU Helper Functions ---
def _update_confusion_matrix(gt_label, pred_label, confusion_matrix):
    gt_flat = gt_label.flatten()
    pred_flat = pred_label.flatten()
    mask = (gt_flat >= 0) & (gt_flat < confusion_matrix.shape[0])
    conf_index = confusion_matrix.shape[0] * gt_flat[mask].astype(int) + pred_flat[mask]
    hist = np.bincount(conf_index, minlength=confusion_matrix.size)
    confusion_matrix += hist.reshape(confusion_matrix.shape)

def _get_metrics_from_confusion_matrix(confusion_matrix):
    fg_confusion_matrix = confusion_matrix[1:, 1:]
    
    fg_correct = np.sum(np.diag(fg_confusion_matrix))
    fg_total = np.sum(fg_confusion_matrix)
    fg_pixel_accuracy = fg_correct / (fg_total + 1e-15)

    intersection = np.diag(fg_confusion_matrix)
    union = np.sum(fg_confusion_matrix, axis=1) + np.sum(fg_confusion_matrix, axis=0) - intersection
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        iou = intersection / union

    mean_iou = np.nanmean(iou)
    
    metrics_dict = {
        "fg_pixel_accuracy": fg_pixel_accuracy,
        "mean_iou": mean_iou
    }
    
    return metrics_dict

def test(net, testloader, device: torch.device):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    
    # 适配 U-Net 获取类别数的方式
    num_classes = net.segmentation_head[0].out_channels
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    net.eval()
    with torch.no_grad():
        for images, masks in tqdm(testloader, desc="Evaluating"):
            images = images.to(device)
            masks_cpu = masks.cpu().numpy().astype(np.int32)
            masks = masks.to(device)
            
            # SMP 模型直接返回张量
            outputs = net(images)
            
            loss = criterion(outputs, masks)
            if not torch.isnan(loss):
                total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.int32)
            _update_confusion_matrix(masks_cpu, preds, confusion_matrix)

    avg_loss = total_loss / len(testloader)
    metrics_dict = _get_metrics_from_confusion_matrix(confusion_matrix)
    
    return avg_loss, metrics_dict