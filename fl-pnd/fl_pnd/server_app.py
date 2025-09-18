"""
fl-pnd: A Flower / PyTorch app for Semantic Segmentation.
This file defines the Flower Server logic and provides a helper function
to create server components for manual simulation startups.
"""
from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig, ServerApp, ServerAppComponents

# 从我们自己的 task.py 中导入正确的模型获取函数
from .task import get_net

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation results using a weighted average."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if sum(examples) == 0:
        return {} # Return empty dict if no examples
    aggregated_accuracy = sum(accuracies) / sum(examples)
    return {"accuracy": aggregated_accuracy}

def get_server_components(num_rounds: int = 3):
    """
    Creates server components (strategy and config).
    """
    print("--- Initializing server strategy and config ---")
    
    net = get_net()
    initial_parameters = [val.cpu().numpy() for _, val in net.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_parameters)

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)
    
    print("--- Server strategy and config initialized ---")
    return ServerAppComponents(strategy=strategy, config=config)

def server_fn(context: fl.common.Context) -> ServerAppComponents:
    """Defines the Flower server components for `flwr run`."""
    # This could be extended to read from context.run_config if needed
    return get_server_components()

app = ServerApp(
    server_fn=server_fn,
)