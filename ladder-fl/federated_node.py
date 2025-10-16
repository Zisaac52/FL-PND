import sys
import os
from typing import List
import torch
import numpy as np

# 为了能够导入 fl-pnd 中的模块，我们需要将其路径添加到 sys.path
# 这是一个临时的解决方案，在更复杂的项目中，会使用更规范的包管理
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .node import Node
from .data_structures import UpperChainBlock, LowerChainBlock, ModelParameters
from fl_pnd.task import get_net, set_parameters, train, test

# 假设数据集和相关配置是可用的
# 在真实场景中，这部分需要更复杂的处理
from fl_pnd.dataset import get_dataloader, load_data, get_val_dataloader
from torchvision.transforms import Compose, ToTensor, Normalize

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"--- FederatedNode will use device: {DEVICE} ---")

class FederatedNode(Node):
    """
    一个实现了真实联邦学习逻辑的节点。
    """
    def __init__(self, node_id: str, network, valloader, class_weights, trainloader=None):
        super().__init__(node_id, network)
        self.model = get_net().to(DEVICE)
        self.trainloader = trainloader
        self.valloader = valloader
        self.class_weights = class_weights # 接收计算好的权重
        self.reputation = 0.1 # 节点的初始信誉值

    def get_model_parameters(self) -> List[np.ndarray]:
        """从本地模型中提取参数，格式为NumPy数组列表"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_model_parameters(self, parameters: List[np.ndarray]):
        """将NumPy数组列表格式的参数加载到本地模型中"""
        set_parameters(self.model, parameters)

    def run_local_round(self, current_round: int) -> dict:
        """执行一轮完整的本地操作：训练+评估"""
        # 训练
        print(f"节点 {self.node_id}: 开始在真实数据上进行本地训练 (Round {current_round})...")
        train(
            net=self.model,
            trainloader=self.trainloader,
            epochs=1,
            device=DEVICE,
            class_weights=self.class_weights,
            current_round=current_round
        )
        print(f"节点 {self.node_id}: 本地训练完成。")

        # 评估
        print(f"节点 {self.node_id}: 开始在验证集上进行评估...")
        loss, metrics = test(net=self.model, testloader=self.valloader, device=DEVICE)
        print(f"节点 {self.node_id}: 评估完成 - Loss: {loss:.4f}, Mean IoU: {metrics['mean_iou']:.4f}, FG Pixel Acc: {metrics['fg_pixel_accuracy']:.4f}")
        return metrics
