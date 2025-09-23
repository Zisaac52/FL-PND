# """
# fl-pnd: A Flower / PyTorch app for Semantic Segmentation.
# This file defines the Flower Server logic and provides a helper function
# to create server components for manual simulation startups.
# """
# from typing import List, Tuple
# import flwr as fl
# from flwr.common import Metrics
# from flwr.server.strategy import FedAvg
# from flwr.server import ServerConfig, ServerApp, ServerAppComponents
# from flwr.server.strategy import FedProx

# # 从我们自己的 task.py 中导入正确的模型获取函数
# from .task import get_net

# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     """
#     A generic weighted average aggregation function that handles a dictionary
#     of metrics.
#     """
#     if not metrics:
#         return {}
        
#     # 聚合所有数值类型的指标
#     aggregated_metrics = {}
#     total_examples = sum([num_examples for num_examples, _ in metrics])

#     # 检查第一个客户端返回的指标字典，获取所有可用的指标键
#     if total_examples > 0 and metrics[0][1]:
#         metric_keys = [key for key, value in metrics[0][1].items() if isinstance(value, (int, float))]
        
#         for key in metric_keys:
#             # 计算该指标的加权总和
#             weighted_sum = sum([num_examples * m[key] for num_examples, m in metrics if key in m])
#             # 计算加权平均
#             aggregated_metrics[key] = weighted_sum / total_examples
            
#     return aggregated_metrics

# def get_server_components(num_rounds: int = 3):
#     """
#     Creates server components (strategy and config).
#     """
#     print("--- Initializing server strategy and config ---")
    
#     net = get_net()
#     initial_parameters = [val.cpu().numpy() for _, val in net.state_dict().items()]
#     initial_parameters = fl.common.ndarrays_to_parameters(initial_parameters)

#     # strategy = FedAvg(
#     #     fraction_fit=1.0,
#     #     fraction_evaluate=1.0,
#     #     min_fit_clients=2,
#     #     min_evaluate_clients=2,
#     #     min_available_clients=2,
#     #     initial_parameters=initial_parameters,
#     #     evaluate_metrics_aggregation_fn=weighted_average,
#     # )

#     strategy = FedProx(
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         min_fit_clients=2,
#         min_evaluate_clients=2,
#         min_available_clients=2,
#         initial_parameters=initial_parameters,
#         evaluate_metrics_aggregation_fn=weighted_average,
#         # 新增 FedProx 的核心参数
#         proximal_mu=0.1 # 近端项的强度，一个需要调整的超参数
#     )
    
#     config = ServerConfig(num_rounds=num_rounds)
    
#     print("--- Server strategy and config initialized ---")
#     return ServerAppComponents(strategy=strategy, config=config)

# def server_fn(context: fl.common.Context) -> ServerAppComponents:
#     """Defines the Flower server components for `flwr run`."""
#     # This could be extended to read from context.run_config if needed
#     return get_server_components()

# app = ServerApp(
#     server_fn=server_fn,
# )

"""
fl-pnd: A Flower / PyTorch app for Semantic Segmentation.
This file defines the Flower Server logic and provides a helper function
to create server components for manual simulation startups.
"""
from typing import List, Tuple, Dict
import flwr as fl
from flwr.common import Metrics, Scalar
from flwr.server.strategy import FedProx # 使用 FedProx
from flwr.server.strategy import FedAvg # 也保留 FedAvg 以便对比
from flwr.server import ServerConfig, ServerApp, ServerAppComponents
from .task import get_net

# --- [新增] ---
def fit_config(server_round: int) -> Dict[str, Scalar]:
    """
    Return training configuration dict for each round.
    This function is used by the strategy to configure the clients.
    We pass the current server round to the clients for two-stage fine-tuning.
    """
    config = {
        "server_round": server_round,
        "proximal_mu": 0.1 # FedProx 的核心参数
    }
    return config

# --- 通用加权平均函数 (保持不变) ---
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # ... (这个函数保持原样)
    if not metrics:
        return {}
    aggregated_metrics = {}
    total_examples = sum([num_examples for num_examples, _ in metrics])
    if total_examples > 0 and metrics[0][1]:
        metric_keys = [key for key, value in metrics[0][1].items() if isinstance(value, (int, float))]
        for key in metric_keys:
            weighted_sum = sum([num_examples * m[key] for num_examples, m in metrics if key in m])
            aggregated_metrics[key] = weighted_sum / total_examples
    return aggregated_metrics


def get_server_components(num_rounds: int = 3):
    """Creates server components (strategy and config)."""
    print("--- Initializing server strategy and config ---")
    
    net = get_net()
    initial_parameters = [val.cpu().numpy() for _, val in net.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_parameters)

    # --- [核心修改] ---
    # 1. 使用 FedProx 策略
    # 2. 注册 on_fit_config_fn 以便向客户端传递配置
    # strategy = FedProx(
    #     fraction_fit=1.0,
    #     fraction_evaluate=1.0,
    #     min_fit_clients=2,
    #     min_evaluate_clients=2,
    #     min_available_clients=2,
    #     initial_parameters=initial_parameters,
    #     evaluate_metrics_aggregation_fn=weighted_average,
    #     on_fit_config_fn=fit_config, # <--- 注册配置函数
    #     proximal_mu=0.1  # <--- 在这里添加这一行
    # )

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config, # <--- 注册配置函数
    )



    config = ServerConfig(num_rounds=num_rounds)
    
    print("--- Server strategy and config initialized ---")
    return ServerAppComponents(strategy=strategy, config=config)

# ... (server_fn 和 app 保持不变) ...
def server_fn(context: fl.common.Context) -> ServerAppComponents:
    return get_server_components()

app = ServerApp(server_fn=server_fn)