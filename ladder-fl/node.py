import json
import copy
from typing import List, Dict
from .data_structures import UpperChainBlock, LowerChainBlock, ModelParameters
from .serde import parameters_to_serializable, serializable_to_parameters

class Node:
    """
    代表网络中的一个客户端/节点。
    """
    def __init__(self, node_id: str, network):
        self.node_id = node_id
        self.network = network  # 网络模拟器，用于广播和接收消息
        self.upper_chain: Dict[str, UpperChainBlock] = {}
        self.lower_chain: Dict[str, LowerChainBlock] = {}
        self.mempool: List[UpperChainBlock] = [] # 待处理的上链区块

    def broadcast_upper_block(self, block: UpperChainBlock):
        """序列化并广播一个新的上链区块到网络"""
        print(f"节点 {self.node_id}: 正在广播上链区块 {block.hash[:6]}...")
        # 创建一个副本以进行序列化，避免修改原始区块对象
        block_copy = copy.deepcopy(block)
        block_copy.model_params = parameters_to_serializable(block_copy.model_params)
        self.network.broadcast(sender_id=self.node_id, message=json.dumps(block_copy.__dict__))

    def broadcast_lower_block(self, block: LowerChainBlock):
        """序列化并广播一个新的下链区块到网络"""
        print(f"节点 {self.node_id}: 正在广播下链区块 {block.hash[:6]}...")
        block_copy = copy.deepcopy(block)
        block_copy.aggregated_global_model = parameters_to_serializable(block_copy.aggregated_global_model)
        self.network.broadcast(sender_id=self.node_id, message=json.dumps(block_copy.__dict__))

    def receive_message(self, sender_id: str, message: str):
        """从网络接收消息"""
        data = json.loads(message)
        
        # 简单的消息路由：判断是上链区块还是下链区块
        if "standard_upper_block_hash" in data:
            self._handle_lower_block(data)
        else:
            self._handle_upper_block(data)

    def _validate_upper_block(self, block: UpperChainBlock, difficulty: int = 2) -> bool:
        """验证一个上链区块的有效性"""
        # 1. 验证哈希
        expected_hash = block.calculate_hash()
        if block.hash != expected_hash:
            print(f"节点 {self.node_id}: [验证失败] 上链区块 {block.hash[:6]} 哈希不匹配。")
            return False
        
        # 2. 验证PoW难度
        prefix = "0" * difficulty
        if not block.hash.startswith(prefix):
            print(f"节点 {self.node_id}: [验证失败] 上链区块 {block.hash[:6]} 未满足PoW难度。")
            return False

        # 3. 验证父区块是否存在 (简化验证)
        if block.parent_lower_hash not in self.lower_chain:
            print(f"节点 {self.node_id}: [验证失败] 上链区块 {block.hash[:6]} 的父下链区块未知。")
            return False
            
        return True

    def _validate_lower_block(self, block: LowerChainBlock) -> bool:
        """验证一个下链区块的有效性"""
        # 1. 验证哈希
        expected_hash = block.calculate_hash()
        if block.hash != expected_hash:
            print(f"节点 {self.node_id}: [验证失败] 下链区块 {block.hash[:6]} 哈希不匹配。")
            return False

        # 2. 验证父区块是否存在
        if block.parent_lower_hash not in self.lower_chain:
            print(f"节点 {self.node_id}: [验证失败] 下链区块 {block.hash[:6]} 的父区块未知。")
            return False

        # 3. 验证引用的上链区块是否存在于mempool中
        mempool_hashes = {b.hash for b in self.mempool}
        referenced_hashes = {block.standard_upper_block_hash} | set(block.forked_upper_block_hashes)
        if not referenced_hashes.issubset(mempool_hashes):
            print(f"节点 {self.node_id}: [验证失败] 下链区块 {block.hash[:6]} 引用了未知的上链区块。")
            return False

        return True

    def _handle_upper_block(self, block_data: dict):
        """处理并反序列化接收到的上链区块"""
        block_data["model_params"] = serializable_to_parameters(block_data["model_params"])
        block = UpperChainBlock(**block_data)
        
        if self._validate_upper_block(block):
            print(f"节点 {self.node_id}: 接收并验证通过了来自 {block.client_id} 的上链区块 {block.hash[:6]}。")
            self.mempool.append(block)
        else:
            print(f"节点 {self.node_id}: 拒绝了一个无效的上链区块 {block.hash[:6]}。")


    def _handle_lower_block(self, block_data: dict):
        """处理并反序列化接收到的下链区块"""
        block_data["aggregated_global_model"] = serializable_to_parameters(block_data["aggregated_global_model"])
        block = LowerChainBlock(**block_data)

        if self._validate_lower_block(block):
            print(f"节点 {self.node_id}: 接收并验证通过了下链区块 (Round {block.round}) {block.hash[:6]}。")
            self.lower_chain[block.hash] = block
            
            # 清理mempool
            confirmed_hashes = {block.standard_upper_block_hash} | set(block.forked_upper_block_hashes)
            original_mempool_size = len(self.mempool)
            self.mempool = [b for b in self.mempool if b.hash not in confirmed_hashes]
            print(f"节点 {self.node_id}: Mempool清理完成，移除了 {original_mempool_size - len(self.mempool)} 个已确认区块。")
        else:
            print(f"节点 {self.node_id}: 拒绝了一个无效的下链区块 {block.hash[:6]}。")
