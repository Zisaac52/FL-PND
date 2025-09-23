"""
fl-pnd: A Flower / PyTorch app for Semantic Segmentation.
This file defines the custom dataset, data loading, and data-related utilities
like class weight calculation.
"""
import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F
from datasets import Dataset as HFDataset, Features, Image as HFImage
from flwr_datasets.partitioner import IidPartitioner
from tqdm import tqdm

# --- 自定义 PyTorch Dataset ---
class PND_Segmentation_Dataset(Dataset):
    def __init__(self, root_dir, image_set='train', transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        
        image_set_file = os.path.join(root_dir, 'ImageSets', 'Segmentation', f'{image_set}.txt')
        self.images_dir = os.path.join(root_dir, 'JPEGImages')
        self.masks_dir = os.path.join(root_dir, 'SegmentationClass')

        with open(image_set_file, 'r') as f:
            self.filenames = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        image_path = os.path.join(self.images_dir, f'{name}.jpg')
        mask_path = os.path.join(self.masks_dir, f'{name}.png')
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
            
        return image, mask

# --- 自定义数据变换 ---
class SegmentationTransforms:
    def __init__(self, size=(512, 512), is_train=True):
        self.size = size
        self.is_train = is_train
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, mask):
        image = F.resize(image, self.size)
        mask = F.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST)

        if self.is_train and torch.rand(1) > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        
        image = F.to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long) # Safer conversion for masks
        image = self.normalize(image)

        return image, mask

# --- Hugging Face & Partitioner 逻辑 (用于训练集) ---
def load_data(num_partitions: int):
    """加载训练集、转换为HF格式并创建Partitioner。"""
    DATASET_ROOT = './Panax notoginseng disease dataset/VOC2007'
    full_train_dataset = PND_Segmentation_Dataset(root_dir=DATASET_ROOT, image_set='train', transforms=None)

    dataset_dict = {"image": [img for img, _ in full_train_dataset], "mask": [mask for _, mask in full_train_dataset]}
    features = Features({'image': HFImage(decode=True), 'mask': HFImage(decode=True)})
    hf_dataset = HFDataset.from_dict(dataset_dict, features=features)

    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = hf_dataset
    
    return partitioner

def get_dataloader(partition, batch_size: int, is_train: bool):
    """从一个Hugging Face数据分区创建一个PyTorch DataLoader。"""
    transforms = SegmentationTransforms(is_train=is_train)
    
    def apply_transforms_hf(batch):
        transformed_pairs = [transforms(img, mask) for img, mask in zip(batch['image'], batch['mask'])]
        batch['image'] = [pair[0] for pair in transformed_pairs]
        batch['mask'] = [pair[1] for pair in transformed_pairs]
        return batch
    
    partition = partition.with_transform(apply_transforms_hf)

    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        return images, masks
        
    return DataLoader(partition, batch_size=batch_size, shuffle=is_train, collate_fn=collate_fn)

# --- [新增] 全局验证集加载器 ---
def get_val_dataloader(batch_size: int):
    """创建一个全局的验证集 DataLoader。"""
    DATASET_ROOT = './Panax notoginseng disease dataset/VOC2007'
    
    # 1. 创建一个 PyTorch Dataset，专门加载验证集
    transforms = SegmentationTransforms(is_train=False)
    val_dataset = PND_Segmentation_Dataset(root_dir=DATASET_ROOT, image_set='val', transforms=transforms)
    
    print(f"全局验证集已加载，包含 {len(val_dataset)} 个样本。")

    # 2. 直接用 PyTorch DataLoader 包装
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- [新增] 类别权重计算函数 ---
def calculate_class_weights(dataset: torch.utils.data.Dataset, num_classes: int) -> torch.Tensor:
    """遍历数据集，计算用于加权交叉熵损失的类别权重。"""
    print("--- Calculating class weights for the training dataset ---")
    class_counts = np.zeros(num_classes)
    
    for _, mask in tqdm(dataset, desc="Counting pixels per class"):
        mask_np = np.array(mask)
        unique, counts = np.unique(mask_np, return_counts=True)
        # 确保 unique 中的值不会超出 class_counts 的索引范围
        valid_indices = unique < num_classes
        class_counts[unique[valid_indices]] += counts[valid_indices]

    print("Total pixel counts per class:", class_counts)

    # 计算权重，为0的类别设置一个极小值防止除零
    total_pixels = np.sum(class_counts)
    weights = total_pixels / (num_classes * (class_counts + 1e-6))
    
    print("Calculated class weights:", weights)
    print("----------------------------------------------------")
    
    return torch.from_numpy(weights).float()