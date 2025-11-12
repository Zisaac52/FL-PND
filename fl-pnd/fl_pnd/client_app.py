"""
fl-pnd: A Flower / PyTorch app for Semantic Segmentation.
This file defines the Flower Client logic.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import time

import flwr as fl
import torch
from flwr.common import Context
from .dataset import get_dataloader
from .task import get_net, set_parameters, train, test
from .data_structures import UpperChainBlock  # <-- 导入区块结构
from .serde import parameters_to_serializable  # <-- 导入序列化工具

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    # ... (__init__, get_parameters, evaluate 方法保持不变) ...
    def __init__(self, cid: str, partitioner, valloader, class_weights: torch.Tensor):
        self.cid = cid
        self.net = get_net().to(DEVICE)
        self.class_weights = class_weights
        
        partition = partitioner.load_partition(int(cid))
        print(f"客户端 {self.cid} 已创建，加载了 {len(partition)} 个训练样本。")
        self.trainloader = get_dataloader(partition, batch_size=4, is_train=True)
        self.valloader = valloader
        self.dataset_size = len(partition)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    # --- [核心修改] ---
    def fit(self, parameters, config):
        """Trains the local model, receiving the current server round from the config."""
        set_parameters(self.net, parameters)
        
        # 从服务器传递过来的 config 字典中获取当前轮数及最新下链哈希
        current_round = config.get("server_round", 0)
        parent_lower_hash = config.get("latest_lower_hash", "GENESIS")
        
        local_epochs = int(config.get("local_epochs", 1))
        start_perf = time.perf_counter()
        start_wall = time.time()
        print(f"--- 客户端 {self.cid} 开始训练 (设备: {DEVICE}, 轮次: {current_round}, epochs: {local_epochs}) ---")
        train(
            net=self.net, 
            trainloader=self.trainloader, 
            epochs=local_epochs,
            device=DEVICE, 
            class_weights=self.class_weights,
            current_round=current_round # <--- 将轮数传递给 train 函数
        )
        
        updated_params = self.get_parameters(config={})

        # 评估本地更新后的模型，以获取最新的 metrics
        loss, metrics = test(net=self.net, testloader=self.valloader, device=DEVICE)
        metrics["loss"] = loss
        training_time = time.perf_counter() - start_perf
        train_finish_ts = time.time()

        # 为服务器准备一个 UpperChainBlock 所需的payload（无 PoW，仅签名/信誉）
        block_payload = {
            "client_id": self.cid,
            "dataset_size": self.dataset_size,
            "parent_lower_hash": parent_lower_hash,
            "metrics": metrics,
            "training_time": training_time,
            "train_finish_ts": train_finish_ts,
        }

        serialized_params_bytes = parameters_to_serializable(updated_params)
        block_payload_json = json.dumps(block_payload)

        metrics_dict = {
            "upper_block_payload_json": block_payload_json,
            "serializable_params_bytes": serialized_params_bytes,
            "local_loss": float(loss),
        }

        return updated_params, self.dataset_size, metrics_dict
        
    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, metrics_dict = test(net=self.net, testloader=self.valloader, device=DEVICE)
        self.last_metrics = metrics_dict # 缓存评估结果，以便 fit 时打包
        return float(loss), len(self.valloader.dataset), metrics_dict

# ... (client_fn_simulation 保持不变) ...
def client_fn_simulation(partitioner, valloader, class_weights: torch.Tensor):
    def client_fn(context: Context) -> fl.client.Client:
        cid = str(context.node_config.get("partition-id", context.node_id))
        return FlowerClient(cid, partitioner, valloader, class_weights).to_client()
    return client_fn
