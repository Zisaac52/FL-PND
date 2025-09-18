# test_ray_gpu.py
import ray
import torch
import os

print(f"[Main Script] Is CUDA available here?         -> {torch.cuda.is_available()}")
print(f"[Main Script] CUDA_VISIBLE_DEVICES env var is -> '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")

# 显式地初始化 Ray，并告诉它集群中有1个GPU
if ray.is_initialized():
    ray.shutdown()
ray.init(num_gpus=1)

@ray.remote(num_gpus=0.5) # 请求0.5个GPU资源来运行这个函数
def check_gpu_in_worker():
    """这个函数将在一个独立的 Ray worker 进程中运行。"""
    print("\n--- INSIDE RAY WORKER ---")
    worker_cuda_available = torch.cuda.is_available()
    print(f"[Ray Worker] Is CUDA available here?      -> {worker_cuda_available}")
    
    # 检查worker进程是否看到了正确的环境变量
    worker_cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f"[Ray Worker] CUDA_VISIBLE_DEVICES env var is -> '{worker_cvd}'")
    
    if not worker_cuda_available:
        raise RuntimeError("Ray worker CANNOT see the GPU!")
        
    return f"Success! Worker sees GPU: {torch.cuda.get_device_name(0)}"

# 异步执行这个远程函数
future = check_gpu_in_worker.remote()

try:
    # 获取结果
    result = ray.get(future)
    print("\n--- RESULT ---")
    print(result)
    print("\n[SUCCESS] Ray is correctly configured to use GPUs in this environment.")
except Exception as e:
    print("\n--- FAILURE ---")
    print(f"An error occurred inside the Ray worker: {e}")
    print("\n[FAILURE] Ray is NOT correctly configured. The problem is with Ray/environment, not Flower.")
finally:
    ray.shutdown()