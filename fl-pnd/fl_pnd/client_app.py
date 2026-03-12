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
from .zkp_groth16 import Groth16Engine, Groth16Error
from .zkp_layers import flatten_selected_arrays, load_layer_whitelist

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
        malicious_config: dict | None = None,
    ):
        self.cid = cid
        self.net = get_net().to(DEVICE)
        self.param_names = list(self.net.state_dict().keys())
        self.class_weights = class_weights
        self.zkp_config = zkp_config or {}
        malicious_cfg = malicious_config or {}
        malicious_clients = {str(cid) for cid in malicious_cfg.get("clients", set())}
        self.malicious_scale = float(malicious_cfg.get("scale", 50.0))
        self.malicious_fraction = float(np.clip(malicious_cfg.get("fraction", 0.3), 0.0, 1.0))
        self.malicious_alpha = float(malicious_cfg.get("alpha", 1.0))
        self.malicious_clip = float(abs(malicious_cfg.get("clip", 1000.0)))
        self.malicious_attack_prob = float(np.clip(malicious_cfg.get("attack_prob", 0.5), 0.0, 1.0))
        self.malicious_mode = str(malicious_cfg.get("mode", "mask_noise")).lower()
        self.full_noise_scale = float(malicious_cfg.get("noise_scale", 1.0))
        raw_rounds = malicious_cfg.get("rounds", set())
        self.malicious_rounds = {int(r) for r in raw_rounds} if raw_rounds else set()
        self.is_malicious = self.cid in malicious_clients
        self._last_attack_stats: dict | None = None
        self._zkp_engine: RoFLL2Engine | None = None
        self._groth16_engine: Groth16Engine | None = None
        self.zkp_scheme = str(self.zkp_config.get("scheme", "none")).lower()
        enabled_flag = bool(self.zkp_config.get("enabled", False))
        self.zkp_enabled = enabled_flag and self.zkp_scheme not in ("none", "")
        self.zkp_range_bits = int(self.zkp_config.get("range_bits", 16))
        self.zkp_partitions = int(self.zkp_config.get("n_partition", 1))
        self.fixed_scale = float(self.zkp_config.get("scale", 1e4))
        self.fixed_clip = self.zkp_config.get("clip")
        self.fixed_tau = float(self.zkp_config.get("tau", 10.0))
        self.groth16_diff_bits = int(self.zkp_config.get("diff_bits", 24))
        self.groth16_bin = self.zkp_config.get("groth16_bin")
        self.groth16_pk = self.zkp_config.get("groth16_pk")
        self.groth16_vk = self.zkp_config.get("groth16_vk")
        self.groth16_layer_prefixes, self.groth16_layer_meta = load_layer_whitelist(self.zkp_config)
        malicious_prefixes, _ = load_layer_whitelist(malicious_cfg)
        # 攻击层优先使用恶意白名单；若未配置则回退到 ZKP 白名单；再为空则攻击全层。
        self.attack_prefixes = list(malicious_prefixes or self.groth16_layer_prefixes)
        
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
        delta_params = self._maybe_make_malicious(delta_params, current_round)
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
        if self._last_attack_stats:
            block_payload["simulated_attack"] = self._last_attack_stats

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

    def _get_groth16_engine(self) -> Groth16Engine | None:
        if not self.zkp_enabled or self.zkp_scheme != "groth16":
            return None
        if self._groth16_engine is None:
            self._groth16_engine = Groth16Engine(
                bin_path=self.groth16_bin,
                pk_path=self.groth16_pk,
                vk_path=self.groth16_vk,
            )
        return self._groth16_engine

    def _groth16_layer_selected(self, name: str) -> bool:
        if not self.groth16_layer_prefixes:
            return False
        return any(name.startswith(prefix) for prefix in self.groth16_layer_prefixes)

    def _maybe_make_malicious(self, delta_params: list[np.ndarray], current_round: int) -> list[np.ndarray]:
        if not self.is_malicious:
            self._last_attack_stats = None
            return delta_params
        rng = np.random.default_rng(seed=(int(time.time() * 1e6) ^ hash((self.cid, current_round))) & 0xFFFFFFFF)
        if self.malicious_attack_prob <= 0.0 or rng.random() > self.malicious_attack_prob:
            self._last_attack_stats = {
                "is_malicious": True,
                "attack": False,
                "round": current_round,
                "mode": self.malicious_mode,
                "reason": "probability",
            }
            return delta_params
        if self.malicious_rounds and current_round not in self.malicious_rounds:
            self._last_attack_stats = {
                "is_malicious": True,
                "attack": False,
                "round": current_round,
                "mode": self.malicious_mode,
                "reason": "round_filter",
            }
            return delta_params

        if self.malicious_mode == "full_noise":
            attacked = []
            total_norm_sq = 0.0
            for original in delta_params:
                noise = rng.standard_normal(size=original.shape).astype(np.float32)
                noise *= self.full_noise_scale
                attacked.append(noise)
                total_norm_sq += float(np.sum(noise ** 2))
            self._last_attack_stats = {
                "is_malicious": True,
                "mode": "full_noise",
                "attack": True,
                "round": current_round,
                "l2_sq": total_norm_sq,
                "noise_scale": self.full_noise_scale,
            }
            print(
                f"[AttackSim] Client {self.cid} round {current_round}: uploaded full Gaussian noise "
                f"(||Δ||^2={total_norm_sq:.2e})."
            )
            return attacked

        prefixes = self.attack_prefixes
        attacked: list[np.ndarray] = []
        total_norm_sq = 0.0
        attacked_layers = 0
        for name, original in zip(self.param_names, delta_params):
            should_attack = not prefixes or any(name.startswith(prefix) for prefix in prefixes)
            if should_attack:
                noise = rng.standard_normal(size=original.shape).astype(np.float32)
                if self.malicious_clip > 0:
                    np.clip(noise, -self.malicious_clip, self.malicious_clip, out=noise)
                mask = rng.random(size=original.shape) < self.malicious_fraction
                perturb = self.malicious_alpha * self.malicious_scale * noise * mask
                attacked_delta = original + perturb
                attacked.append(attacked_delta)
                total_norm_sq += float(np.sum((attacked_delta - original) ** 2))
                attacked_layers += 1
            else:
                attacked.append(original)

        if attacked_layers == 0:
            self._last_attack_stats = {
                "is_malicious": True,
                "l2_sq": 0.0,
                "scale": 0.0,
                "note": "no prefixes matched",
                "mode": "mask_noise",
                "round": current_round,
            }
            return delta_params

        self._last_attack_stats = {
            "is_malicious": True,
            "l2_sq": total_norm_sq,
            "scale": self.malicious_scale,
            "alpha": self.malicious_alpha,
            "fraction": self.malicious_fraction,
            "layers": attacked_layers,
            "round": current_round,
            "mode": "mask_noise",
        }
        print(
            f"[AttackSim] Client {self.cid} round {current_round}: injected malicious delta "
            f"(layers={attacked_layers}, ||Δ||^2={total_norm_sq:.2e})."
        )
        return attacked

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
        if self.zkp_scheme == "groth16":
            if not self.groth16_layer_prefixes:
                print(f"[ZKP] Client {self.cid}: groth16 layer whitelist is empty; skipping proof.")
                return None
            try:
                engine = self._get_groth16_engine()
            except (Groth16Error, FileNotFoundError, OSError) as exc:
                print(f"[ZKP] Client {self.cid}: failed to init Groth16 engine ({exc})")
                return None
            if not engine:
                return None
            flat_vec, matched_names, selected_arrays = flatten_selected_arrays(
                self.param_names,
                delta_params_payload,
                self.groth16_layer_prefixes,
            )
            if flat_vec.size == 0 or not selected_arrays:
                print(f"[ZKP] Client {self.cid}: no parameters matched Groth16 whitelist; skipping proof.")
                return None
            selection_bytes = flat_vec.tobytes()
            selection_hash = compute_payload_hash(selection_bytes)
            try:
                payload = engine.prove_delta(
                    selected_arrays,
                    scale=self.fixed_scale,
                    tau=self.fixed_tau,
                    clip=self.fixed_clip,
                    diff_bits=self.groth16_diff_bits,
                    delta_hash=delta_sha,
                )
            except Groth16Error as exc:
                print(f"[ZKP] Client {self.cid}: Groth16 proof failed ({exc})")
                return None
            payload["selection_hash"] = selection_hash
            payload["selection_len"] = int(flat_vec.size)
            payload["selection_names"] = matched_names
            payload["selection_prefixes"] = list(self.groth16_layer_prefixes)
            return payload
        return None


def client_fn_simulation(
    partitioner,
    valloader,
    class_weights: torch.Tensor,
    zkp_config: dict | None = None,
    malicious_config: dict | None = None,
):
    def client_fn(context: Context) -> fl.client.Client:
        cid = str(context.node_config.get("partition-id", context.node_id))
        return FlowerClient(
            cid,
            partitioner,
            valloader,
            class_weights,
            zkp_config=zkp_config,
            malicious_config=malicious_config,
        ).to_client()

    return client_fn
