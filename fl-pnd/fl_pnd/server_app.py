"""
fl-pnd: Flower server components with Ladder-inspired DAG coordination.
"""
from __future__ import annotations

import json
import math
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import flwr as fl
from flwr.common import Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from .task import get_net
from .data_structures import LowerChainBlock, UpperChainBlock
from .serde import serializable_to_parameters

SERVER_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_NUM_CLIENTS = 10
EMA_ALPHA = 0.3


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics using example counts."""
    if not metrics:
        return {}
    aggregated_metrics: Dict[str, float] = {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0 or not metrics[0][1]:
        return aggregated_metrics
    metric_keys = [
        key for key, value in metrics[0][1].items() if isinstance(value, (int, float))
    ]
    for key in metric_keys:
        weighted_sum = sum(
            num_examples * m[key] for num_examples, m in metrics if key in m
        )
        aggregated_metrics[key] = weighted_sum / total_examples
    return aggregated_metrics


def federated_average(blocks: List[UpperChainBlock]) -> List[np.ndarray]:
    """Classic FedAvg over the collected UpperChainBlocks."""
    if not blocks:
        return []

    weights = torch.tensor(
        [b.dataset_size for b in blocks], dtype=torch.float32, device=SERVER_DEVICE
    )
    weights = weights / weights.sum()

    aggregated: List[np.ndarray] = []
    num_layers = len(blocks[0].model_params)
    for idx in range(num_layers):
        stacked = torch.stack(
            [torch.from_numpy(b.model_params[idx]).to(SERVER_DEVICE) for b in blocks],
            dim=0,
        )
        view_shape = (-1,) + (1,) * (stacked.dim() - 1)
        weighted_sum = torch.sum(stacked * weights.view(view_shape), dim=0)
        aggregated.append(weighted_sum.cpu().numpy())
    return aggregated


class LadderStrategy(fl.server.strategy.FedAvg):
    """FedAvg-compatible strategy which also tracks Ladder DAG metadata."""

    def __init__(
        self,
        num_clients: int = DEFAULT_NUM_CLIENTS,
        local_epochs: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.local_epochs = local_epochs
        self.reputation_ledger = {
            str(i): 1.0 / num_clients for i in range(num_clients)
        }
        self.ema_alpha = EMA_ALPHA
        self.round_start_perf: Dict[int, float] = {}
        self.round_start_wall: Dict[int, float] = {}
        self.round_chain_metrics: List[Dict[str, float]] = []

        temp_net = get_net()
        initial_params = [val.cpu().numpy() for _, val in temp_net.state_dict().items()]
        self.latest_lower_block = LowerChainBlock(
            round=0,
            standard_upper_block_hash="GENESIS",
            forked_upper_block_hashes=[],
            aggregated_global_model=initial_params,
            parent_lower_hash="NULL",
        )
        self.latest_lower_block.hash = self.latest_lower_block.calculate_hash()

        # 每轮向客户端下发 server_round + latest_lower_hash
        self.on_fit_config_fn = self._fit_config

    def _fit_config(self, server_round: int) -> Dict[str, int]:
        self.round_start_perf[server_round] = time.perf_counter()
        self.round_start_wall[server_round] = time.time()
        return {
            "server_round": server_round,
            "latest_lower_hash": self.latest_lower_block.hash,
            "local_epochs": self.local_epochs,
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures,
    ):
        round_start_ts = self.round_start_perf.pop(server_round, None)
        round_start_wall = self.round_start_wall.pop(server_round, None)
        print(f"\n--- [Round {server_round}] Ladder/DAG aggregation start ---")
        received_blocks: List[UpperChainBlock] = []
        for _, fit_res in results:
            payload_json = fit_res.metrics.get("upper_block_payload_json")
            params_bytes = fit_res.metrics.get("serializable_params_bytes")
            if not payload_json or params_bytes is None:
                continue
            payload = json.loads(payload_json)
            params = serializable_to_parameters(params_bytes)
            block = UpperChainBlock(
                client_id=payload["client_id"],
                dataset_size=payload["dataset_size"],
                parent_lower_hash=payload["parent_lower_hash"],
                metrics=payload["metrics"],
                model_params=params,
                training_time=payload.get("training_time", 0.0),
                train_finish_ts=payload.get("train_finish_ts", 0.0),
            )
            block.hash = block.calculate_hash()
            received_blocks.append(block)
            print(
                f"    · UpperChainBlock {block.hash[:6]} "
                f"(client={block.client_id}, parent_lower={block.parent_lower_hash[:6]})"
            )

        if not received_blocks:
            # Fallback to default FedAvg aggregation.
            print("    · No valid blocks received, falling back to FedAvg.")
            return super().aggregate_fit(server_round, results, failures)

        aggregated_params_np = federated_average(received_blocks)
        aggregated_params = fl.common.ndarrays_to_parameters(aggregated_params_np)

        # Reputation update based on reported loss (no PoW).
        for block in received_blocks:
            client_id = block.client_id
            if client_id not in self.reputation_ledger:
                self.reputation_ledger[client_id] = 1.0 / self.num_clients
            loss = block.metrics.get("loss", 1.0)
            quality = 1.0 / (1.0 + loss)
            old_rep = self.reputation_ledger[client_id]
            self.reputation_ledger[client_id] = (
                self.ema_alpha * quality + (1 - self.ema_alpha) * old_rep
            )

        total_rep = sum(self.reputation_ledger.values())
        if total_rep > 0:
            for cid in self.reputation_ledger:
                self.reputation_ledger[cid] /= total_rep

        convergence_node_id = random.choices(
            population=list(self.reputation_ledger.keys()),
            weights=list(self.reputation_ledger.values()),
            k=1,
        )[0]
        print(
            f"    · PoS elected convergence node {convergence_node_id} "
            f"(rep={self.reputation_ledger[convergence_node_id]:.4f})"
        )

        standard_block = next(
            (b for b in received_blocks if b.client_id == convergence_node_id),
            received_blocks[0],
        )
        forked_blocks = [
            b for b in received_blocks if b.client_id != standard_block.client_id
        ]

        self.latest_lower_block = LowerChainBlock(
            round=server_round,
            standard_upper_block_hash=standard_block.hash,
            forked_upper_block_hashes=[b.hash for b in forked_blocks],
            aggregated_global_model=aggregated_params_np,
            parent_lower_hash=self.latest_lower_block.hash,
        )
        self.latest_lower_block.hash = self.latest_lower_block.calculate_hash()
        print(
            f"    · LowerChainBlock {self.latest_lower_block.hash[:6]} created "
            f"(round={server_round}, parent={self.latest_lower_block.parent_lower_hash[:6]})"
        )

        upload_bytes = 0
        training_latency = float("nan")
        consensus_latency = float("nan")
        if received_blocks:
            upload_bytes = sum(
                sum(arr.nbytes for arr in block.model_params) for block in received_blocks
            )
            finish_times = [
                ts for ts in (block.train_finish_ts for block in received_blocks) if ts
            ]
            if finish_times and round_start_wall:
                training_latency = max(finish_times) - round_start_wall
            else:
                training_latency = max(
                    (block.training_time for block in received_blocks),
                    default=float("nan"),
                )

        latency = (
            time.perf_counter() - round_start_ts
            if round_start_ts is not None
            else float("nan")
        )
        throughput = (
            len(received_blocks) / latency if latency and latency > 0 else float("nan")
        )
        if (
            isinstance(latency, (int, float))
            and latency == latency
            and isinstance(training_latency, (int, float))
            and not math.isnan(training_latency)
        ):
            consensus_latency = max(latency - training_latency, 0.0)
        else:
            consensus_latency = float("nan")
        self.round_chain_metrics.append(
            {
                "round": server_round,
                "latency": latency,
                "training_latency": training_latency,
                "consensus_latency": consensus_latency,
                "throughput": throughput,
                "consensus_throughput": (
                    len(received_blocks) / consensus_latency
                    if consensus_latency and consensus_latency > 0
                    else float("nan")
                ),
                "upload_mb": upload_bytes / (1024 * 1024),
                "num_blocks": len(received_blocks),
                "forks": len(forked_blocks),
            }
        )

        return aggregated_params, {}


def get_server_components(
    num_rounds: int = 3,
    num_clients: int = DEFAULT_NUM_CLIENTS,
    eval_fraction: float = 0.3,
    fit_fraction: float = 1.0,
    local_epochs: int = 1,
) -> ServerAppComponents:
    """Prepare Ladder-aware FedAvg strategy plus config."""
    print("--- Initializing server strategy and config ---")

    net = get_net()
    initial_parameters = [val.cpu().numpy() for _, val in net.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_parameters)

    eval_fraction = max(0.0, min(1.0, eval_fraction))
    min_eval_clients = max(1, int(num_clients * eval_fraction)) if eval_fraction > 0 else 0

    fit_fraction = max(0.0, min(1.0, fit_fraction))
    min_fit_clients = max(1, int(num_clients * fit_fraction)) if fit_fraction > 0 else 0

    strategy = LadderStrategy(
        num_clients=num_clients,
        local_epochs=local_epochs,
        fraction_fit=fit_fraction,
        fraction_evaluate=eval_fraction,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_eval_clients,
        min_available_clients=max(1, max(min_fit_clients, min_eval_clients)),
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)
    print("--- Server strategy and config initialized ---")
    return ServerAppComponents(strategy=strategy, config=config)


def server_fn(context: fl.common.Context) -> ServerAppComponents:
    return get_server_components()


app = ServerApp(server_fn=server_fn)
