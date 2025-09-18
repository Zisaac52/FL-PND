"""
Main script to manually start the Flower simulation for the PND project.
This script provides full control over the Ray initialization and
federated learning setup, bypassing `flwr run`.
"""
import flwr as fl
import ray

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

    # 1. 初始化 Ray，并强制指定GPU资源
    print("--- Manually initializing Ray with GPU support ---")
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_gpus=1)
    print("--- Ray Initialized ---")

    # 2. 准备所有数据相关资源
    NUM_CLIENTS = 10
    
    # 2a. 创建用于训练数据分区的 Actor
    print(f"--- Creating DatasetActor for {NUM_CLIENTS} clients ---")
    dataset_actor = DatasetActor.remote(NUM_CLIENTS)
    partitioner = ray.get(dataset_actor.get_partitioner.remote())
    
    # 2b. 创建所有客户端共享的全局验证集 DataLoader
    print("--- Creating global validation dataloader ---")
    valloader = get_val_dataloader(batch_size=8)
    
    # 2c. 计算用于处理类别不平衡的权重
    full_train_dataset = PND_Segmentation_Dataset(
        root_dir='./Panax notoginseng disease dataset/VOC2007', 
        image_set='train'
    )
    num_classes = get_net().classifier[4].out_channels
    class_weights = calculate_class_weights(full_train_dataset, num_classes)

    # 3. 准备客户端工厂函数 (client_fn)，注入所有资源
    client_fn = client_fn_simulation(
        partitioner=partitioner, 
        valloader=valloader, 
        class_weights=class_weights
    )

    # 4. 获取服务器组件 (Strategy 和 ServerConfig)
    #    在这里设置您想要的训练轮数
    server_components = get_server_components(num_rounds=20)

    # 5. 定义客户端所需的计算资源
    client_resources = {"num_cpus": 2, "num_gpus": 0.5}

    # 6. 启动联邦学习仿真
    print("--- Starting Flower Simulation ---")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=server_components.config,
        strategy=server_components.strategy,
        client_resources=client_resources,
    )

    print("\n--- Simulation Finished ---")
    print("Final run history:", history)

    # 7. 关闭 Ray
    ray.shutdown()