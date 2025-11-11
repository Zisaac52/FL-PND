from typing import List
import numpy as np
import pickle


def parameters_to_serializable(parameters: List[np.ndarray]) -> bytes:
    """
    将NumPy数组列表直接序列化为bytes，以便高效传输。
    """
    return pickle.dumps(parameters)


def serializable_to_parameters(serializable_params: bytes) -> List[np.ndarray]:
    """
    将bytes反序列化回NumPy数组列表。
    """
    return pickle.loads(serializable_params)
