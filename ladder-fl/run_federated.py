import random
import time
import numpy as np
import multiprocessing
from multiprocessing import Pool
from typing import List
import copy

from .network import Network
from .federated_node import FederatedNode, test, DEVICE
# 导入所有需要的 dataset 工具
from fl_pnd.dataset import load_data, get_dataloader, get_val_dataloader, PND_Segmentation_Dataset, calculate_class_weights
from fl_pnd.task import get_net # 需要它来获取类别数
from .data_structures import UpperChainBlock, LowerChainBlock
from .serde import serializable_to_parameters, parameters_to_serializable

# --- 真实的联邦平均聚合函数 ---

def federated_average(updates: List[UpperChainBlock]) -> List[np.ndarray]:
    """
    对来自多个客户端的模型更新执行联邦平均。
    """
    print("收敛节点: 正在执行真实的联邦平均...")
    
    # 提取所有模型参数和数据集大小
    all_params = [u.model_params for u in updates]
    dataset_sizes = [u.dataset_size for u in updates]
    total_size = sum(dataset_sizes)

    # 初始化一个新的模型参数列表用于存放聚合结果
    aggregated_params: List[np.ndarray] = []

    # 遍历模型中的每一层参数
    for i in range(len(all_params[0])):
        # 计算该层参数的加权平均
        weighted_sum = sum(params[i] * size for params, size in zip(all_params, dataset_sizes))
        aggregated_layer = weighted_sum / total_size
        aggregated_params.append(aggregated_layer)
        
    return aggregated_params

# --- PoW 辅助函数 ---

def perform_pow(block: UpperChainBlock, difficulty: int) -> UpperChainBlock:
    """模拟工作量证明"""
    prefix = "0" * difficulty
    while not block.hash.startswith(prefix):
        block.nonce += 1
        block.hash = block.calculate_hash()
    print(f"节点 {block.client_id}: PoW完成! Nonce={block.nonce}, Hash={block.hash[:8]}...")
    return block

# --- 仿真主流程 ---

def node_process(args):
    """每个节点运行的独立进程，并返回结果"""
    node_id, partition, valloader, class_weights, lower_block_dict, current_round = args

    # 反序列化下链区块
    lower_block = LowerChainBlock(**lower_block_dict)
    lower_block.aggregated_global_model = serializable_to_parameters(lower_block.aggregated_global_model)

    # 在子进程中创建DataLoader
    trainloader = get_dataloader(partition, batch_size=4, is_train=True)

    # [修改] 将 class_weights 传递给节点
    node = FederatedNode(node_id=node_id, network=None, trainloader=trainloader, valloader=valloader, class_weights=class_weights)
    node.set_model_parameters(lower_block.aggregated_global_model)
    metrics = node.run_local_round(current_round=current_round)
    updated_params = node.get_model_parameters()
    
    upper_block = UpperChainBlock(
        client_id=node.node_id,
        dataset_size=len(node.trainloader.dataset),
        model_params=updated_params,
        metrics=metrics,
        parent_upper_hash="GENESIS_UPPER", # 简化处理
        parent_lower_hash=lower_block.hash
    )
    upper_block.hash = upper_block.calculate_hash()
    upper_block = perform_pow(upper_block, difficulty=2)
    
    # 为了能跨进程返回，将区块对象转换为可序列化的字典
    block_dict = copy.deepcopy(upper_block).__dict__
    block_dict["model_params"] = parameters_to_serializable(upper_block.model_params)
    print(f"节点 {node_id} 完成工作。")
    return block_dict


def run_federated_simulation():
    # ================== 可调整参数 ==================
    NUM_NODES = 10
    NUM_ROUNDS = 5
    MAX_CONCURRENT_WORKERS = 4
    # ==============================================

    print("--- 开始完整并行的真实LadderFL仿真 ---")
    
    # 1. 在主进程中一次性加载、划分和计算所有数据相关资源
    print("--- 正在主进程中准备所有数据资源... ---")
    partitioner = load_data(num_partitions=NUM_NODES)
    partitions = [partitioner.load_partition(i) for i in range(NUM_NODES)]
    valloader = get_val_dataloader(batch_size=16)
    
    # 确保路径相对于项目根目录
    full_train_dataset = PND_Segmentation_Dataset(
        root_dir='/root/PND/Panax notoginseng disease dataset/VOC2007',
        image_set='train'
    )
    num_classes = get_net().segmentation_head[0].out_channels
    class_weights = calculate_class_weights(full_train_dataset, num_classes)
    print("--- 数据资源准备完成 ---")

    # 2. 创建创世区块
    temp_node = FederatedNode(node_id="temp", network=None, valloader=valloader, class_weights=class_weights)
    initial_params = temp_node.get_model_parameters()
    del temp_node
    
    latest_lower_block = LowerChainBlock(
        round=0,
        standard_upper_block_hash="GENESIS_UPPER",
        forked_upper_block_hashes=[],
        aggregated_global_model=initial_params,
        parent_lower_hash="NULL"
    )
    latest_lower_block.hash = latest_lower_block.calculate_hash()
    
    # --- 主训练循环 ---
    global_metrics_history = []
    for r in range(1, NUM_ROUNDS + 1):
        print(f"\n{'='*20} 第 {r} 轮开始 (基于区块 {latest_lower_block.hash[:6]}) {'='*20}")

        # 为了跨进程传递，序列化模型参数
        lower_block_dict = copy.deepcopy(latest_lower_block).__dict__
        lower_block_dict["aggregated_global_model"] = parameters_to_serializable(latest_lower_block.aggregated_global_model)

        # ... [并行训练部分保持不变] ...
        # 3. 创建进程池并分发任务 (现在传递partition而不是partition_id)
        tasks = [(f"client_{i}", partitions[i], valloader, class_weights, lower_block_dict, r) for i in range(NUM_NODES)]
        
        with Pool(processes=MAX_CONCURRENT_WORKERS) as pool:
            collected_blocks_data = pool.map(node_process, tasks)

        print(f"\n--- 第 {r} 轮：所有节点已并行完成训练和PoW ---")
        
        # ... [聚合部分保持不变] ...
        collected_blocks = []
        for data in collected_blocks_data:
            data["model_params"] = serializable_to_parameters(data["model_params"])
            collected_blocks.append(UpperChainBlock(**data))
        print(f"\n--- 第 {r} 轮：所有节点的本地评估结果 ---")
        for block in collected_blocks:
            print(f"  - {block.client_id}: Mean IoU = {block.metrics.get('mean_iou', 0):.4f}, FG Acc = {block.metrics.get('fg_pixel_accuracy', 0):.4f}")
        print("---------------------------------")
        standard_block = collected_blocks[0]
        forked_blocks = collected_blocks[1:]
        new_global_model_params = federated_average(collected_blocks)
        
        # 5. 创建新的下链区块
        latest_lower_block = LowerChainBlock(
            round=r,
            standard_upper_block_hash=standard_block.hash,
            forked_upper_block_hashes=[b.hash for b in forked_blocks],
            aggregated_global_model=new_global_model_params,
            parent_lower_hash=latest_lower_block.hash
        )
        latest_lower_block.hash = latest_lower_block.calculate_hash()

        print(f"\n--- 第 {r} 轮结束 ---")
        print(f"新的全局模型已在主进程中成功聚合。")

        # --- [核心修改] 在每轮结束后评估全局模型 ---
        print(f"\n--- 正在评估第 {r} 轮聚合后的全局模型性能 ---")
        eval_node = FederatedNode(node_id="eval_node", network=None, valloader=valloader, class_weights=class_weights)
        eval_node.set_model_parameters(new_global_model_params)
        loss, metrics = test(net=eval_node.model, testloader=eval_node.valloader, device=DEVICE)
        metrics["loss"] = loss
        global_metrics_history.append(metrics)
        print(f" >> 第 {r} 轮全局模型评估结果: Mean IoU = {metrics['mean_iou']:.4f}, Loss = {loss:.4f}")

    # --- 在所有轮次结束后，打印最终总结 ---
    print("\n\n" + "="*30 + " 最终仿真结果总结 " + "="*30)
    print(f"总轮次: {NUM_ROUNDS}, 客户端数量: {NUM_NODES}")
    print("\n全局模型性能演进:")
    print("----------------------------------------------------------")
    print("| Round |    Loss    |  Mean IoU  | FG Pixel Acc |")
    print("----------------------------------------------------------")
    for i, metrics in enumerate(global_metrics_history):
        print(f"|   {i+1}   |  {metrics['loss']:.4f}  |  {metrics['mean_iou']:.4f}  |    {metrics['fg_pixel_accuracy']:.4f}    |")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run_federated_simulation()