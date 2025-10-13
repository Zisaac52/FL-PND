from typing import List
import numpy as np

def parameters_to_serializable(parameters: List[np.ndarray]) -> List[list]:
    """
    将NumPy数组列表转换为可JSON序列化的Python列表。
    """
    return [p.tolist() for p in parameters]

def serializable_to_parameters(serializable_params: List[list]) -> List[np.ndarray]:
    """
    将Python列表转换回NumPy数组列表。
    """
    return [np.array(p) for p in serializable_params]
