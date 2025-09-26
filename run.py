# run.py (位于 PND/ 目录下)

import flwr as fl
import ray
import json # 用于读取 ABI 和地址文件
import hashlib
from web3 import Web3

# 从您的项目中导入所有需要的函数/类
from fl_pnd.client_app import client_fn_simulation
from fl_pnd.server_app import get_server_components
from fl_pnd.dataset import (
    load_data, 
    get_val_dataloader, 
    PND_Segmentation_Dataset, 
    calculate_class_weights
)
from fl_pnd.task import get_net

@ray.remote(num_cpus=1)
class DatasetActor:
    """A Ray Actor to load and hold the dataset partitioner once."""
    def __init__(self, num_partitions: int):
        print("DatasetActor: Loading and partitioning dataset...")
        self.partitioner = load_data(num_partitions)
        print("DatasetActor: Dataset loaded and partitioned.")
    
    def get_partitioner(self):
        return self.partitioner

# --- 脚本主逻辑 ---
if __name__ == "__main__":
    # --- [区块链设置] ---
    # 1. 加载合约信息
    with open("fl-pnd/fl_pnd/contracts/contract-address.json", "r") as f:
        CONTRACT_ADDRESS = json.load(f)["address"]
    with open("fl-pnd/fl_pnd/contracts/FederatedLearning.json", "r") as f:
        CONTRACT_ABI = json.load(f)["abi"]
    
    # 2. 连接到 Ganache
    w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
    contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
    server_account = w3.eth.accounts[0] # 使用 Ganache 的第一个账户作为服务器
    w3.eth.default_account = server_account
    print(f"成功连接到智能合约，地址: {CONTRACT_ADDRESS}")

    # --- [联邦学习设置] ---
    print("--- Manually initializing Ray with GPU support ---")
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_gpus=1)
    print("--- Ray Initialized ---")

    NUM_CLIENTS = 10
    
    dataset_actor = DatasetActor.remote(NUM_CLIENTS)
    partitioner = ray.get(dataset_actor.get_partitioner.remote())
    valloader = get_val_dataloader(batch_size=8)
    
    full_train_dataset = PND_Segmentation_Dataset(
        root_dir='./Panax notoginseng disease dataset/VOC2007', 
        image_set='train'
    )
    temp_net = get_net()
    num_classes = temp_net.segmentation_head[0].out_channels
    del temp_net
    class_weights = calculate_class_weights(full_train_dataset, num_classes)

    # 将合约信息注入到客户端工厂函数
    client_fn = client_fn_simulation(
        partitioner=partitioner, 
        valloader=valloader, 
        class_weights=class_weights,
        contract_address=CONTRACT_ADDRESS,
        contract_abi=CONTRACT_ABI
    )

    # 获取服务器策略，并包装 aggregate_fit 方法
    server_components = get_server_components(num_rounds=5) # 先用10轮测试
    strategy = server_components.strategy
    
    original_aggregate_fit = strategy.aggregate_fit
    def blockchain_aggregate_fit(server_round, results, failures):
        aggregated_params_tuple = original_aggregate_fit(server_round, results, failures)
        if aggregated_params_tuple is None:
            return None
        
        aggregated_params, _ = aggregated_params_tuple
        
        agg_params_np = fl.common.parameters_to_ndarrays(aggregated_params)
        agg_params_bytes = b"".join([param.tobytes() for param in agg_params_np])
        agg_hash = hashlib.sha256(agg_params_bytes).digest()

        print(f"服务器正在向区块链敲定第 {server_round} 轮...")
        try:
            tx_hash = contract.functions.finalizeRound(agg_hash).transact()
            w3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"服务器成功敲定第 {server_round} 轮！")
        except Exception as e:
            print(f"服务器敲定轮次失败: {e}")
            
        return aggregated_params_tuple
    
    strategy.aggregate_fit = blockchain_aggregate_fit
    
    client_resources = {"num_cpus": 2, "num_gpus": 0.5}

    print("--- Starting Flower Simulation with Blockchain Integration ---")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=server_components.config,
        strategy=strategy,
        client_resources=client_resources,
    )

    print("\n--- Simulation Finished ---")
    print("Final run history:", history)
    ray.shutdown()