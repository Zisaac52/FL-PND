from __future__ import annotations

"""
fl-pnd: A Flower / PyTorch app for Semantic Segmentation.
This file defines the Flower Client logic.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import json
import time
import numpy as np

import flwr as fl
import torch
from flwr.common import Context
from .dataset import get_dataloader
from .task import get_net, set_parameters, train, test
from .data_structures import UpperChainBlock  # <-- 导入区块结构
from .serde import parameters_to_serializable  # <-- 导入序列化工具
from .zkp_bindings import (
    RoFLL2Engine,
    L2ProofArtifacts,
    compute_payload_hash,
    ZKPBindingError,
)
from .zkp_fixed_l2 import build_certificate as build_fixed_l2_certificate

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    # ... (__init__, get_parameters, evaluate 方法保持不变) ...
    def __init__(
        self,
        cid: str,
        partitioner,
        valloader,
        class_weights: torch.Tensor,
        zkp_config: dict | None = None,
    ):
        self.cid = cid
        self.net = get_net().to(DEVICE)
        self.class_weights = class_weights
        self.zkp_config = zkp_config or {}
        self._zkp_engine: RoFLL2Engine | None = None
        self.zkp_scheme = str(self.zkp_config.get("scheme", "none")).lower()
        enabled_flag = bool(self.zkp_config.get("enabled", False))
        self.zkp_enabled = enabled_flag and self.zkp_scheme not in ("none", "")
        self.zkp_range_bits = int(self.zkp_config.get("range_bits", 16))
        self.zkp_partitions = int(self.zkp_config.get("n_partition", 1))
        self.fixed_scale = float(self.zkp_config.get("scale", 1e4))
        self.fixed_clip = self.zkp_config.get("clip")
        self.fixed_tau = float(self.zkp_config.get("tau", 10.0))
        
        partition = partitioner.load_partition(int(cid))
        print(f"客户端 {self.cid} 已创建，加载了 {len(partition)} 个训练样本。")
        self.trainloader = get_dataloader(partition, batch_size=4, is_train=True)
        self.valloader = valloader
        self.dataset_size = len(partition)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    # --- [核心修改] ---
    def fit(self, parameters, config):
        """Trains the local model, receiving the current server round from the config."""
        set_parameters(self.net, parameters)
        initial_params = [np.array(p, copy=True) for p in parameters]

        # 从服务器传递过来的 config 字典中获取当前轮数及最新下链哈希
        current_round = config.get("server_round", 0)
        parent_lower_hash = config.get("latest_lower_hash", "GENESIS")
        
        local_epochs = int(config.get("local_epochs", 1))
        start_perf = time.perf_counter()
        print(f"--- 客户端 {self.cid} 开始训练 (设备: {DEVICE}, 轮次: {current_round}, epochs: {local_epochs}) ---")
        avg_train_loss = train(
            net=self.net, 
            trainloader=self.trainloader, 
            epochs=local_epochs,
            device=DEVICE, 
            class_weights=self.class_weights,
            current_round=current_round # <--- 将轮数传递给 train 函数
        )
        
        updated_params = self.get_parameters(config={})
        metrics = {"loss": float(avg_train_loss)}

        delta_params = [
            updated.astype(np.float32) - base.astype(np.float32)
            for updated, base in zip(updated_params, initial_params)
        ]
        delta_params_fp16 = [delta.astype(np.float16) for delta in delta_params]
        serialized_params_bytes = parameters_to_serializable(
            delta_params_fp16, dtype="float16"
        )
        model_size_bytes = len(serialized_params_bytes)
        training_time = time.perf_counter() - start_perf
        train_finish_ts = time.time()
        delta_sha = compute_payload_hash(serialized_params_bytes)

        # 为服务器准备一个 UpperChainBlock 所需的payload（无 PoW，仅签名/信誉）
        block_payload = {
            "client_id": self.cid,
            "dataset_size": self.dataset_size,
            "parent_lower_hash": parent_lower_hash,
            "metrics": metrics,
            "training_time": training_time,
            "train_finish_ts": train_finish_ts,
            "payload_type": "delta",
            "compression": "float16",
            "model_size_bytes": model_size_bytes,
            "delta_sha256": delta_sha,
        }
        zkp_payload = self._build_zkp_payload(
            delta_params_float32=delta_params,
            delta_params_payload=delta_params_fp16,
            serialized_bytes=serialized_params_bytes,
            delta_sha=delta_sha,
        )
        if zkp_payload:
            block_payload["zkp"] = zkp_payload

        block_payload_json = json.dumps(block_payload)

        metrics_dict = {
            "upper_block_payload_json": block_payload_json,
            "serializable_params_bytes": serialized_params_bytes,
            "local_loss": float(avg_train_loss),
            "zkp_enabled": float(bool(zkp_payload)),
        }

        return updated_params, self.dataset_size, metrics_dict
        
    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, metrics_dict = test(net=self.net, testloader=self.valloader, device=DEVICE)
        self.last_metrics = metrics_dict # 缓存评估结果，以便 fit 时打包
        return float(loss), len(self.valloader.dataset), metrics_dict

# ... (client_fn_simulation 保持不变) ...
    def _get_zkp_engine(self) -> RoFLL2Engine | None:
        if not self.zkp_enabled or self.zkp_scheme != "rofl":
            return None
        if self._zkp_engine is None:
            lib_path = self.zkp_config.get("lib_path")
            self._zkp_engine = RoFLL2Engine(lib_path)
        return self._zkp_engine

    def _build_zkp_payload(self, delta_params_float32, delta_params_payload, serialized_bytes, delta_sha):
        if not self.zkp_enabled:
            return None
        if self.zkp_scheme == "rofl":
            try:
                engine = self._get_zkp_engine()
                proof = (
                    engine.prove_l2_bound(
                        delta_params_float32,
                        range_bits=self.zkp_range_bits,
                        n_partition=self.zkp_partitions,
                    )
                    if engine
                    else None
                )
            except (ZKPBindingError, FileNotFoundError, OSError) as exc:
                print(f"[ZKP] Client {self.cid}: failed to generate RoFL proof: {exc}")
                return None
            if proof:
                return {
                    "scheme": "rofl_l2",
                    "artifacts": proof.to_payload(),
                    "vector_hash": delta_sha,
                }
            return None
        if self.zkp_scheme in {"fixed", "fixed_l2"}:
            start = time.perf_counter()
            try:
                certificate = build_fixed_l2_certificate(
                    delta_params_payload,
                    scale=self.fixed_scale,
                    clip=self.fixed_clip,
                    tau=self.fixed_tau,
                )
                prove_time = time.perf_counter() - start
                print(
                    f"[ZKP] Client {self.cid}: fixed_l2 certificate "
                    f"l2_sq={certificate['l2_sq']} tau_sq={certificate['tau_sq']}"
                )
            except ValueError as exc:
                print(f"[ZKP] Client {self.cid}: fixed-point certificate rejected ({exc})")
                return None
            certificate["float_hash"] = delta_sha
            cert_bytes = json.dumps(certificate).encode("utf-8")
            certificate["prove_time"] = prove_time
            certificate["proof_bytes"] = len(cert_bytes)
            return {"scheme": "fixed_l2", "certificate": certificate}
        return None


def client_fn_simulation(partitioner, valloader, class_weights: torch.Tensor, zkp_config: dict | None = None):
    def client_fn(context: Context) -> fl.client.Client:
        cid = str(context.node_config.get("partition-id", context.node_id))
        return FlowerClient(cid, partitioner, valloader, class_weights, zkp_config=zkp_config).to_client()
    return client_fn
