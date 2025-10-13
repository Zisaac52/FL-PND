import random
import time
from .network import Network
from .node import Node
from .data_structures import UpperChainBlock, LowerChainBlock

# --- 模拟的辅助函数 ---

def simulate_training(node_id: str, global_model: dict) -> dict:
    """模拟客户端的本地训练过程"""
    print(f"节点 {node_id}: 正在基于全局模型进行本地训练...")
    time.sleep(random.uniform(0.5, 1.5)) # 模拟训练耗时
    # 简单地返回一个随机修改的模型作为更新
    trained_model = global_model.copy()
    trained_model["accuracy"] = random.uniform(0.6, 0.9)
    return trained_model

def perform_pow(block: UpperChainBlock, difficulty: int) -> UpperChainBlock:
    """模拟工作量证明"""
    prefix = "0" * difficulty
    while not block.hash.startswith(prefix):
        block.nonce += 1
        block.hash = block.calculate_hash()
    print(f"节点 {block.client_id}: PoW完成! Nonce={block.nonce}, Hash={block.hash[:8]}...")
    return block

def aggregate_models(blocks: list[UpperChainBlock]) -> dict:
    """模拟FedAvg聚合"""
    print("收敛节点: 正在聚合模型...")
    total_size = sum(b.dataset_size for b in blocks)
    # 简单地取准确率的加权平均
    avg_accuracy = sum(b.model_params["accuracy"] * b.dataset_size for b in blocks) / total_size
    return {"accuracy": avg_accuracy}


# --- 仿真主流程 ---

def run_simulation():
    print("--- 开始LadderFL仿真 ---")
    
    # 1. 初始化网络和节点
    network = Network()
    nodes = [Node(node_id=f"client_{i}", network=network) for i in range(3)]
    for node in nodes:
        network.join(node)

    # 2. 创建创世区块 (由client_0手动创建)
    genesis_model = {"accuracy": 0.5}
    genesis_lower_block = LowerChainBlock(
        round=0,
        standard_upper_block_hash="GENESIS_UPPER",
        forked_upper_block_hashes=[],
        aggregated_global_model=genesis_model,
        parent_lower_hash="NULL"
    )
    genesis_lower_block.hash = genesis_lower_block.calculate_hash()
    
    # 所有节点都接收并存储创世区块
    for node in nodes:
        node.lower_chain[genesis_lower_block.hash] = genesis_lower_block
    
    print(f"\n--- 第 1 轮开始 (基于创世区块 {genesis_lower_block.hash[:6]}) ---")

    # 3. 所有节点进行本地训练并广播上链区块
    for node in nodes:
        local_model = simulate_training(node.node_id, genesis_model)
        upper_block = UpperChainBlock(
            client_id=node.node_id,
            dataset_size=random.randint(100, 500),
            model_params=local_model,
            parent_upper_hash="GENESIS_UPPER", # 简化处理
            parent_lower_hash=genesis_lower_block.hash
        )
        upper_block.hash = upper_block.calculate_hash()
        # 执行PoW
        upper_block = perform_pow(upper_block, difficulty=2)
        node.broadcast_upper_block(upper_block)

    # 4. 选出收敛节点并进行聚合
    # 根据规则，上一轮的标准区块创建者是本轮的收敛节点。
    # 在这里我们手动指定 client_1 作为下一轮的收敛节点来简化流程。
    convergence_node = nodes[1]
    print(f"\n--- {convergence_node.node_id} 被选为收敛节点 ---")
    
    # 收敛节点从自己的mempool中收集区块进行聚合
    collected_blocks = convergence_node.mempool
    if not collected_blocks:
        print("错误：收敛节点没有收集到任何上链区块！")
        return

    # 选出标准区块 (这里简化为第一个)
    standard_block = collected_blocks[0]
    forked_blocks = collected_blocks[1:]

    new_global_model = aggregate_models(collected_blocks)
    
    # 5. 创建并广播新的下链区块
    new_lower_block = LowerChainBlock(
        round=1,
        standard_upper_block_hash=standard_block.hash,
        forked_upper_block_hashes=[b.hash for b in forked_blocks],
        aggregated_global_model=new_global_model,
        parent_lower_hash=genesis_lower_block.hash
    )
    new_lower_block.hash = new_lower_block.calculate_hash()
    convergence_node.broadcast_lower_block(new_lower_block)

    print(f"\n--- 第 1 轮结束 ---")
    print(f"新的全局模型: {new_global_model}")
    print(f"网络中最后的下链区块 (Round 1): {nodes[0].lower_chain[new_lower_block.hash]}")

if __name__ == "__main__":
    run_simulation()
