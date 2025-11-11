from dataclasses import dataclass, field
from typing import List, Any, Dict

# 为了简化，模型参数暂时用字典表示
ModelParameters = Dict[str, Any]

@dataclass
class UpperChainBlock:
    """
    上链区块，由客户端在本地训练后生成。
    """
    # --- 无默认值的字段 ---
    client_id: str
    dataset_size: int
    model_params: ModelParameters
    parent_lower_hash: str
    model_size_bytes: int = 0
    
    # --- 有默认值的字段 ---
    metrics: Dict[str, float] = field(default_factory=dict)
    hash: str = ""

    def _get_hashable_string(self) -> str:
        """创建一个用于哈希计算的、确定性的字符串表示"""
        # 我们只包含核心的、不可变的数据
        # 使用json.dumps并对键进行排序，以确保每次输出都相同
        import json
        payload = {
            "client_id": self.client_id,
            "dataset_size": self.dataset_size,
            "parent_lower_hash": self.parent_lower_hash
        }
        return json.dumps(payload, sort_keys=True)

    def calculate_hash(self) -> str:
        """计算区块的哈希值"""
        import hashlib
        block_string = self._get_hashable_string()
        return hashlib.sha256(block_string.encode()).hexdigest()

@dataclass
class LowerChainBlock:
    """
    下链区块，由收敛节点生成，用于聚合模型和同步网络。
    """
    round: int
    standard_upper_block_hash: str
    forked_upper_block_hashes: List[str]
    aggregated_global_model: ModelParameters
    parent_lower_hash: str
    # 区块自身的哈希
    hash: str = ""

    def _get_hashable_string(self) -> str:
        """创建一个用于哈希计算的、确定性的字符串表示"""
        import json
        payload = {
            "round": self.round,
            "standard_upper_block_hash": self.standard_upper_block_hash,
            "forked_upper_block_hashes": sorted(self.forked_upper_block_hashes), # 排序以保证确定性
            "parent_lower_hash": self.parent_lower_hash
        }
        return json.dumps(payload, sort_keys=True)

    def calculate_hash(self) -> str:
        """计算区块的哈希值"""
        import hashlib
        block_string = self._get_hashable_string()
        return hashlib.sha256(block_string.encode()).hexdigest()
