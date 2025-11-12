from typing import List
import numpy as np
import pickle


def parameters_to_serializable(parameters: List[np.ndarray], dtype: str = "float32") -> bytes:
    """将NumPy数组列表序列化为bytes，支持可选的dtype压缩。"""
    arrays = []
    for param in parameters:
        arr = np.array(param, copy=True)
        if dtype != "float32":
            arr = arr.astype(dtype)
        arrays.append(arr)
    payload = {"dtype": dtype, "arrays": arrays}
    return pickle.dumps(payload)


def serializable_to_parameters(serializable_params: bytes) -> List[np.ndarray]:
    """从bytes还原数组，并转换回float32。"""
    data = pickle.loads(serializable_params)
    dtype = data.get("dtype", "float32")
    arrays = data["arrays"]
    if dtype != "float32":
        arrays = [arr.astype(np.float32) for arr in arrays]
    return arrays
