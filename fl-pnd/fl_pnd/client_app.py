"""
fl-pnd: A Flower / PyTorch app for Semantic Segmentation.
This file defines the Flower Client logic.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import flwr as fl
import torch
from flwr.common import Context
from .dataset import get_dataloader
from .task import get_net, set_parameters, train, test

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, partitioner, valloader, class_weights: torch.Tensor):
        self.cid = cid
        self.net = get_net().to(DEVICE)
        self.class_weights = class_weights
        
        partition = partitioner.load_partition(int(cid))
        print(f"客户端 {self.cid} 已创建，加载了 {len(partition)} 个训练样本。")
        self.trainloader = get_dataloader(partition, batch_size=4, is_train=True)
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        
        print(f"--- 客户端 {self.cid} 开始训练 (设备: {DEVICE}) ---")
        train(
            net=self.net, 
            trainloader=self.trainloader, 
            epochs=2, 
            device=DEVICE, 
            class_weights=self.class_weights
        )
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(net=self.net, testloader=self.valloader, device=DEVICE)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


def client_fn_simulation(partitioner, valloader, class_weights: torch.Tensor):
    def client_fn(context: Context) -> fl.client.Client:
        cid = str(context.node_config.get("partition-id", context.node_id))
        return FlowerClient(cid, partitioner, valloader, class_weights).to_client()
    return client_fn