import random
import time
import numpy as np
import torch
import multiprocessing
from multiprocessing import Pool
from typing import List
import copy
import os
import gc

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
    对来自多个客户端的模型更新执行联邦平均 (在GPU上执行以加速)。
    """
    print("收敛节点: 正在执行真实的联邦平均 (GPU加速)...")
    
    if not updates:
        return []

    # 提取所有模型参数和数据集大小
    all_params_np = [u.model_params for u in updates]
    dataset_sizes = [u.dataset_size for u in updates]
    total_size = sum(dataset_sizes)

    # 初始化一个新的模型参数列表用于存放聚合结果
    aggregated_params_np: List[np.ndarray] = []

    # 将权重转换为PyTorch张量
    weights = torch.tensor(dataset_sizes, dtype=torch.float32, device=DEVICE) / total_size

    # 遍历模型中的每一层参数
    for i in range(len(all_params_np[0])):
        # 将这一层所有客户端的参数收集起来，并转换为GPU上的张量
        layer_params = [torch.from_numpy(params[i]).to(DEVICE) for params in all_params_np]
        
        # 使用堆叠和加权求和来高效计算
        stacked_params = torch.stack(layer_params, dim=0)
        
        # 调整权重张量的形状以进行广播
        view_shape = (-1,) + (1,) * (stacked_params.dim() - 1)
        weighted_sum = torch.sum(stacked_params * weights.view(view_shape), dim=0)
        
        aggregated_params_np.append(weighted_sum.cpu().numpy())
        
    return aggregated_params_np

def get_params_size_in_bytes(params: List[np.ndarray]) -> int:
    """计算模型参数列表的总字节大小"""
    return sum(p.nbytes for p in params)

def node_process(args):
    """
    一个无状态的工作进程函数。每次调用都会创建一个新的节点实例来执行任务。
    """
    node_id, partition, valloader, class_weights, lower_block_dict, current_round = args

    # 在子进程中创建 DataLoader
    trainloader = get_dataloader(partition, batch_size=4, is_train=True)

    # 创建节点实例
    node = FederatedNode(
        node_id=node_id,
        network=None,
        trainloader=trainloader,
        valloader=valloader,
        class_weights=class_weights
    )

    # 反序列化下链区块以获取新的全局模型
    lower_block = LowerChainBlock(**lower_block_dict)
    lower_block.aggregated_global_model = serializable_to_parameters(lower_block.aggregated_global_model)

    # 设置模型参数并运行本地轮次
    node.set_model_parameters(lower_block.aggregated_global_model)
    metrics = node.run_local_round(current_round=current_round)
    
    # 获取损失和更新后的参数
    loss, _ = test(net=node.model, testloader=node.valloader, device=DEVICE)
    metrics['loss'] = loss
    updated_params = node.get_model_parameters()
    
    # 创建并准备要返回的上链区块
    upper_block = UpperChainBlock(
        client_id=node.node_id,
        dataset_size=len(node.trainloader.dataset),
        model_params=updated_params,
        metrics=metrics,
        parent_lower_hash=lower_block.hash,
        model_size_bytes=get_params_size_in_bytes(updated_params)
    )
    upper_block.hash = upper_block.calculate_hash()
    
    # 为了能跨进程返回，将区块对象转换为可序列化的字典
    block_dict = copy.deepcopy(upper_block).__dict__
    block_dict["model_params"] = parameters_to_serializable(upper_block.model_params)
    print(f"节点 {node_id} 完成工作。")
    return block_dict


def run_federated_simulation():
    # ================== 可调整参数 ==================
    NUM_NODES = 10
    NUM_ROUNDS = 3
    MAX_CONCURRENT_WORKERS = 4
    EMA_ALPHA = 0.3 # 指数移动平均的alpha值
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

    # 3. 初始化信誉账本
    reputation_ledger = {f"client_{i}": 1.0 / NUM_NODES for i in range(NUM_NODES)}
    print(f"\n--- 初始信誉分布 ---")
    for cid, rep in reputation_ledger.items():
        print(f"  - {cid}: {rep:.4f}")
    print("----------------------")
    
    # --- 主训练循环 ---
    global_metrics_history = []
    system_metrics_history = []
    
    # 创建一个持久化的进程池，在所有轮次中重复使用
    with Pool(processes=MAX_CONCURRENT_WORKERS) as pool:
        for r in range(1, NUM_ROUNDS + 1):
            print(f"\n{'='*20} 第 {r} 轮开始 (基于区块 {latest_lower_block.hash[:6]}) {'='*20}")
            round_start_time = time.time()

            # 为了跨进程传递，序列化模型参数
            lower_block_dict = copy.deepcopy(latest_lower_block).__dict__
            lower_block_dict["aggregated_global_model"] = parameters_to_serializable(latest_lower_block.aggregated_global_model)

            # 3. 创建任务列表
            tasks = [(f"client_{i}", partitions[i], valloader, class_weights, lower_block_dict, r) for i in range(NUM_NODES)]
            
            # 使用持久化的进程池执行任务
            collected_blocks_data = pool.map(node_process, tasks)

            print(f"\n--- 第 {r} 轮：所有节点已并行完成本地训练 ---")
            
            # 4. 反序列化所有收集到的上链区块
            collected_blocks = []
            for data in collected_blocks_data:
                data["model_params"] = serializable_to_parameters(data["model_params"])
                collected_blocks.append(UpperChainBlock(**data))
            
            # 5. PoS委员会选举：根据信誉选出一个收敛节点
            client_ids = list(reputation_ledger.keys())
            reputations = list(reputation_ledger.values())
            
            convergence_node_id = random.choices(client_ids, weights=reputations, k=1)[0]
            print(f"\n--- PoS选举结果 ---")
            print(f"节点 {convergence_node_id} 被选为本轮的收敛节点 (信誉: {reputation_ledger[convergence_node_id]:.4f})")
            print("--------------------")

            # 6. 由收敛节点进行聚合
            # 在这个模拟中，我们直接在主进程中完成聚合，但标记它是由谁完成的
            print(f"收敛节点 {convergence_node_id}: 开始聚合模型...")
            # 简单起见，我们假设收敛节点总是诚实的，并选择第一个区块作为标准
            standard_block = next(b for b in collected_blocks if b.client_id == convergence_node_id)
            forked_blocks = [b for b in collected_blocks if b.client_id != convergence_node_id]
            new_global_model_params = federated_average(collected_blocks)
            print(f"收敛节点 {convergence_node_id}: 模型聚合完成。")

            # 7. 信誉更新 (对所有参与节点)
            print(f"\n--- 主进程 (模拟委员会) 更新信誉 ---")
            for block in collected_blocks:
                client_id = block.client_id
                loss = block.metrics.get('loss', 10.0) # 如果没有loss，给予高惩罚
                quality_score = 1.0 / (1.0 + loss)
                
                old_reputation = reputation_ledger[client_id]
                reputation_ledger[client_id] = (EMA_ALPHA * quality_score) + (1 - EMA_ALPHA) * old_reputation

            # 归一化声誉
            total_reputation = sum(reputation_ledger.values())
            for cid in reputation_ledger:
                reputation_ledger[cid] /= total_reputation
            
            print("更新后信誉分布:")
            for cid, rep in sorted(reputation_ledger.items(), key=lambda item: item[1], reverse=True):
                 print(f"    - {cid}: {rep:.4f}")
            print("--------------------------")

            # 8. 创建新的下链区块
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

            # --- [新增] 系统性能指标计算 ---
            round_end_time = time.time()
            round_duration = round_end_time - round_start_time
            throughput = NUM_NODES / round_duration
            total_upload_bytes = sum(b.model_size_bytes for b in collected_blocks)
            
            system_metrics = {
                "latency": round_duration,
                "throughput": throughput,
                "upload_mb": total_upload_bytes / (1024 * 1024)
            }
            system_metrics_history.append(system_metrics)
            print(f" >> 第 {r} 轮系统性能: Latency = {round_duration:.2f}s, Throughput = {throughput:.2f} updates/sec, Upload = {system_metrics['upload_mb']:.2f} MB")

    # --- 在所有轮次结束后，打印最终总结 ---
    print("\n\n" + "="*30 + " 最终仿真结果总结 " + "="*30)
    print(f"总轮次: {NUM_ROUNDS}, 客户端数量: {NUM_NODES}")
    print("\n全局模型性能演进:")
    print("-------------------------------------------------------------------------------------------------")
    print("| Round |    Loss    |  Mean IoU  | FG Pixel Acc | Latency (s) | Throughput (ups/s) | Upload (MB) |")
    print("-------------------------------------------------------------------------------------------------")
    for i, (ml_metrics, sys_metrics) in enumerate(zip(global_metrics_history, system_metrics_history)):
        print(f"|   {i+1}   |  {ml_metrics['loss']:.4f}  |  {ml_metrics['mean_iou']:.4f}  |    {ml_metrics['fg_pixel_accuracy']:.4f}    | {sys_metrics['latency']:^11.2f} | {sys_metrics['throughput']:^18.2f} | {sys_metrics['upload_mb']:^11.2f} |")
    print("-------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run_federated_simulation()