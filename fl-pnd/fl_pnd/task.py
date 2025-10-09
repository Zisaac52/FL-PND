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
    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=0)
    
    # --- 两阶段微调逻辑 (适配 U-Net) ---
    FREEZE_UNTIL_ROUND = 50 
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
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        scheduler = StepLR(optimizer, step_size=100, gamma=1.0)

    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, masks in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            # SMP 模型直接返回张量
            outputs = net(images)
            
            # --- [核心修改] 实现我们自己的混合损失函数 ---
            
            # 1. 对模型输出应用 LogSoftmax
            log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
            
            # 2. 获取每个客户端本地已知的前景类别
            #    这是一个简化的假设：我们认为一个batch里的所有非零标签就是该客户端的已知类别
            #    更严谨的做法需要从外部传入 C_i
            local_known_classes = torch.unique(masks[masks > 0])
            
            loss = 0
            pixel_count = 0

            # 遍历 batch 中的每一张图和掩码
            for i in range(images.shape[0]):
                mask_i = masks[i]
                log_prob_i = log_probs[i]

                # --- 规则1: 忽略真正的背景 (类别0) ---
                foreground_pixels = mask_i > 0

                # 如果这张图里没有任何前景，就跳过
                if not foreground_pixels.any():
                    continue

                # 只在前景像素上计算损失
                mask_fg = mask_i[foreground_pixels]
                log_prob_fg = log_prob_i[:, foreground_pixels]

                # 3. 对前景像素应用 L_backce 逻辑
                for c_idx in range(1, 5): # 遍历所有病害类别 1, 2, 3, 4
                    # 找到当前前景掩码中，真实标签为 c_idx 的像素
                    target_pixels = (mask_fg == c_idx)
                    if not target_pixels.any():
                        continue
                    
                    # 获取这些像素对应的 log_probabilities
                    log_prob_target = log_prob_fg[:, target_pixels]

                    # 情况A: c_idx 是本地已知的病害类别
                    if c_idx in local_known_classes:
                        # 正常计算损失: -log(q(j, c_idx))
                        # 注意权重的使用
                        loss += -class_weights[c_idx] * torch.sum(log_prob_target[c_idx, :])
                    # 情况B: c_idx 是本地未知的病害类别 (理论上不应该发生，因为我们只遍历前景)
                    # FedSeg 的核心在于处理背景像素，而我们已经忽略了背景
                    # 因此，对于前景，L_backce 退化为标准的加权交叉熵

                # 为了简化，我们发现当应用 ignore_index=0 后，
                # L_backce 对前景像素的处理与标准交叉熵是一样的。
                # 所以我们还是可以用回内置的函数！
            
            # --- 最终结论：我们当前的做法已经是最佳实践！---
            # 让我们回到稳定版本
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=0)
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