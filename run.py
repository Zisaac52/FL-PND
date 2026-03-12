"""
Main script to manually start the Flower simulation for the PND project.
This script provides full control over the Ray initialization and
federated learning setup, bypassing `flwr run`.
"""
import argparse
from pathlib import Path

import flwr as fl
import ray

from fl_pnd.client_app import client_fn_simulation
from fl_pnd.server_app import get_server_components
from fl_pnd.dataset import (
    load_data,
    get_val_dataloader,
    PND_Segmentation_Dataset,
    calculate_class_weights,
)
from fl_pnd.task import get_net


def _format_metric_dict(entries):
    """Convert list of (round, value) tuples to {round: float(value)}."""
    if not entries:
        return {}
    return {rnd: float(val) for rnd, val in entries}


def _extract_eval_metrics(history):
    """Normalize distributed evaluation metrics into per-metric dicts."""
    metrics_distributed = getattr(history, "metrics_distributed", {})
    eval_section = None
    if isinstance(metrics_distributed, dict):
        eval_section = metrics_distributed.get("evaluate")
        if eval_section is None and metrics_distributed:
            # Some Flower versions store evaluation metrics directly at the top level.
            eval_section = metrics_distributed
    if not eval_section:
        return {}

    # Case 1: Flower already aggregated per metric (dict of lists)
    if isinstance(eval_section, dict):
        out = {}
        for metric_name, entries in eval_section.items():
            out[metric_name] = _format_metric_dict(entries)
        return out

    # Case 2: Flower kept a list of (round, metrics_dict)
    if isinstance(eval_section, list):
        out = {}
        for rnd, metrics in eval_section:
            for metric_name, value in metrics.items():
                out.setdefault(metric_name, {})[rnd] = float(value)
        return out

    return {}


def print_history_summary(history, total_rounds: int, num_clients: int, chain_metrics=None):
    """Render a compact table similar to ladder-fl/run_federated.py output."""
    losses = {}
    if history.losses_distributed:
        losses = {rnd: float(loss) for rnd, loss in history.losses_distributed}

    metrics_eval = _extract_eval_metrics(history)
    mean_iou = metrics_eval.get("mean_iou", {})
    fg_acc = metrics_eval.get("fg_pixel_accuracy", {})

    rounds = sorted(losses.keys())
    if not rounds:
        rounds = list(range(1, total_rounds + 1))

    line = "-" * 89
    print("\n" + "=" * 30 + " 最终仿真结果总结 " + "=" * 30)
    print(f"总轮次: {total_rounds}, 客户端数量: {num_clients}\n")
    print("全局模型性能演进:")
    print(line)

    if chain_metrics:
        print("\n区块链性能指标:")
        line_chain = "-" * 191
        print(line_chain)
        print("| Round | Latency (s) | Training Latency (s) | Consensus Latency (s) | Other Latency (s) | Throughput (blocks/s) | Consensus Throughput (blocks/s) | Upload (MB) | Blocks | Forks | ZKP Avg L2^2 | ZKP Max L2^2 | ZKP Rejected | ZK.Prove (s) | ZK.Verify (s) | Proof Size (KB) |")
        print(line_chain)
        for entry in chain_metrics:
            print(
                f"|{entry['round']:>4}   | {entry['latency']:11.2f} |"
                f" {entry.get('training_latency', float('nan')):21.2f} |"
                f" {entry.get('consensus_latency', float('nan')):22.2f} |"
                f" {entry.get('other_latency', float('nan')):19.2f} |"
                f" {entry['throughput']:23.2f} |"
                f" {entry.get('consensus_throughput', float('nan')):32.2f} |"
                f" {entry['upload_mb']:11.2f} |"
                f" {entry['num_blocks']:6} | {entry['forks']:5} |"
                f" {entry.get('zkp_l2_avg', float('nan')):13.2f} |"
                f" {entry.get('zkp_l2_max', float('nan')):13.2f} |"
                f" {entry.get('zkp_rejected', 0):13} |"
                f" {entry.get('zkp_prove_avg', float('nan')):12.2f} |"
                f" {entry.get('zkp_verify_avg', float('nan')):13.2f} |"
                f" {entry.get('zkp_proof_kb', float('nan')):16.2f} |"
            )
        print(line_chain)
    elif chain_metrics is not None:
        print("\n区块链性能指标: 无有效区块（可能因缺少/无效 ZKP 被跳过，已回退到 FedAvg）。")
    print("| Round |    Loss    |  Mean IoU  | FG Pixel Acc |")
    print(line)
    for rnd in rounds:
        loss_val = losses.get(rnd, float("nan"))
        miou_val = mean_iou.get(rnd, float("nan"))
        fg_val = fg_acc.get(rnd, float("nan"))
        print(
            f"|{rnd:>4}   | {loss_val:8.4f} | {miou_val:8.4f} | {fg_val:12.4f} |"
        )
    print(line)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Flower + Ladder simulation.")
    parser.add_argument("--num-clients", type=int, default=10, help="Total number of virtual clients.")
    parser.add_argument("--num-rounds", type=int, default=3, help="Number of federated rounds.")
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Number of local epochs each selected client should run per round.",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.3,
        help="Fraction of clients sampled for evaluation each round.",
    )
    parser.add_argument(
        "--fit-fraction",
        type=float,
        default=1.0,
        help="Fraction of clients sampled for training each round.",
    )
    parser.add_argument(
        "--enable-zkp",
        action="store_true",
        help="Require clients to attach ZKP/certificates (scheme controlled by --zkp-scheme).",
    )
    parser.add_argument(
        "--zkp-lib",
        type=str,
        default="fl-pnd/zkp/librofl_crypto.so",
        help="Path to librofl_crypto.so (only used when --enable-zkp).",
    )
    parser.add_argument(
        "--zkp-range-bits",
        type=int,
        default=16,
        help="Bit-length bound for the RoFL L2 proof (symmetric range).",
    )
    parser.add_argument(
        "--zkp-partitions",
        type=int,
        default=1,
        help="Number of Pedersen partitions to use when constructing ZKP commitments.",
    )
    parser.add_argument(
        "--zkp-scheme",
        choices=["rofl", "fixed", "groth16", "none"],
        default="rofl",
        help="Choose the ZKP backend: RoFL Bulletproofs, lightweight fixed-point, Groth16, or none.",
    )
    parser.add_argument(
        "--zkp-scale",
        type=float,
        default=1e2,
        help="Fixed-point scale factor for the lightweight scheme.",
    )
    parser.add_argument(
        "--zkp-clip",
        type=float,
        default=None,
        help="Optional clipping bound applied before fixed-point encoding.",
    )
    parser.add_argument(
        "--zkp-tau",
        type=float,
        default=10.0,
        help="L2 bound (in float space) for the lightweight scheme.",
    )
    parser.add_argument(
        "--groth16-bin",
        type=str,
        default="zkp-groth16-l2/target/release/main",
        help="Path to the Groth16 CLI binary.",
    )
    parser.add_argument(
        "--groth16-pk",
        type=str,
        default=None,
        help="Path to the Groth16 proving key (required for clients).",
    )
    parser.add_argument(
        "--groth16-vk",
        type=str,
        default=None,
        help="Path to the Groth16 verifying key (required for the server).",
    )
    parser.add_argument(
        "--malicious-clients",
        type=str,
        default="",
        help="Comma-separated client ids to simulate as malicious (e.g. 1,4).",
    )
    parser.add_argument(
        "--malicious-scale",
        type=float,
        default=50.0,
        help="Std-dev multiplier for malicious noise (larger => stronger attack).",
    )
    parser.add_argument(
        "--malicious-fraction",
        type=float,
        default=0.3,
        help="Portion of parameters (0-1) within targeted layers to perturb.",
    )
    parser.add_argument(
        "--malicious-alpha",
        type=float,
        default=1.0,
        help="Strength of perturbation mixed with original delta (0 keeps original).",
    )
    parser.add_argument(
        "--malicious-clip",
        type=float,
        default=1000.0,
        help="Clip bound applied to generated malicious noise (abs value).",
    )
    parser.add_argument(
        "--malicious-attack-prob",
        type=float,
        default=0.5,
        help="Probability (0-1) that a malicious client attacks in a given round.",
    )
    parser.add_argument(
        "--malicious-mode",
        type=str,
        choices=["mask_noise", "full_noise"],
        default="mask_noise",
        help="Attack mode: partial mask noise or full random delta replacement.",
    )
    parser.add_argument(
        "--malicious-noise-scale",
        type=float,
        default=1.0,
        help="Std-dev of Gaussian noise when using full_noise mode.",
    )
    parser.add_argument(
        "--malicious-layer-whitelist",
        type=str,
        default="",
        help="Optional JSON whitelist for malicious attack layers (same format as Groth16 whitelist).",
    )
    parser.add_argument(
        "--malicious-rounds",
        type=str,
        default="",
        help="Optional comma-separated rounds where malicious attacks are allowed (e.g. 5,10,20).",
    )
    parser.add_argument(
        "--groth16-diff-bits",
        type=int,
        default=256,
        help="Bit-length for the integer difference gadget used in Groth16 witness generation.",
    )
    parser.add_argument(
        "--groth16-layer-whitelist",
        type=str,
        default="doc/groth16_whitelist.json",
        help="JSON file listing parameter name prefixes to include in Groth16 proofs.",
    )
    args = parser.parse_args()

    if args.enable_zkp and args.zkp_scheme.lower() == "groth16":
        # The current Groth16 `prove` path is hard-wired to DEFAULT_DIFF_BITS in Rust.
        # Keep CLI aligned to avoid witness/prove circuit mismatch.
        if args.groth16_diff_bits != 256:
            parser.error(
                "--groth16-diff-bits must be 256 for the current Groth16 binary "
                "(otherwise proof generation fails with AssignmentMissing)."
            )

        bin_path = Path(args.groth16_bin)
        if not bin_path.exists():
            fallback_bin = Path("zkp-groth16-l2/target/release/main")
            if bin_path.name == "zkp-groth16-l2" and fallback_bin.exists():
                print(
                    f"[Config] Groth16 binary not found at {bin_path}, "
                    f"falling back to {fallback_bin}."
                )
                args.groth16_bin = str(fallback_bin)
            else:
                parser.error(
                    f"--groth16-bin not found: {bin_path}. "
                    "Build the binary first or pass the correct path."
                )

        if not args.groth16_pk:
            parser.error("--groth16-pk is required when --enable-zkp and --zkp-scheme groth16.")
        if not Path(args.groth16_pk).exists():
            parser.error(f"--groth16-pk not found: {args.groth16_pk}")

        if not args.groth16_vk:
            parser.error("--groth16-vk is required when --enable-zkp and --zkp-scheme groth16.")
        if not Path(args.groth16_vk).exists():
            parser.error(f"--groth16-vk not found: {args.groth16_vk}")

    return args


@ray.remote(num_cpus=1)
class DatasetActor:
    """A Ray Actor to load and hold the dataset partitioner once."""
    def __init__(self, num_partitions: int):
        print("DatasetActor: Loading and partitioning dataset...")
        self.partitioner = load_data(num_partitions)
        print("DatasetActor: Dataset loaded and partitioned.")
    
    def get_partitioner(self):
        return self.partitioner

# --- 脚本主逻辑 ---
if __name__ == "__main__":

    args = parse_args()
    # 1. 初始化 Ray，并强制指定GPU资源
    print("--- Manually initializing Ray with GPU support ---")
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_gpus=1)
    print("--- Ray Initialized ---")

    # 2. 准备所有数据相关资源
    NUM_CLIENTS = args.num_clients
    
    # 2a. 创建用于训练数据分区的 Actor
    print(f"--- Creating DatasetActor for {NUM_CLIENTS} clients ---")
    dataset_actor = DatasetActor.remote(NUM_CLIENTS)
    partitioner = ray.get(dataset_actor.get_partitioner.remote())
    
    # 2b. 创建所有客户端共享的全局验证集 DataLoader
    print("--- Creating global validation dataloader ---")
    valloader = get_val_dataloader(batch_size=8)
    
    # 2c. 计算用于处理类别不平衡的权重
    full_train_dataset = PND_Segmentation_Dataset(
        root_dir='./Panax notoginseng disease dataset/VOC2007', 
        image_set='train'
    )
    # num_classes = get_net().classifier[4].out_channels
    # SMP U-Net's output layer is called 'segmentation_head'
    temp_net = get_net()
    num_classes = temp_net.segmentation_head[0].out_channels
    del temp_net # Free up memory
    
    class_weights = calculate_class_weights(full_train_dataset, num_classes)

    # 3. 准备客户端工厂函数 (client_fn)，注入所有资源
    zkp_config = None
    if args.enable_zkp:
        scheme = args.zkp_scheme.lower()
        zkp_config = {
            "enabled": True,
            "scheme": scheme,
        }
        if scheme == "rofl":
            zkp_config.update(
                {
                    "lib_path": args.zkp_lib,
                    "range_bits": max(1, args.zkp_range_bits),
                    "n_partition": max(1, args.zkp_partitions),
                }
            )
        elif scheme == "fixed":
            zkp_config.update(
                {
                    "scale": args.zkp_scale,
                    "clip": args.zkp_clip,
                    "tau": args.zkp_tau,
                }
            )
        elif scheme == "groth16":
            zkp_config.update(
                {
                    "scale": args.zkp_scale,
                    "clip": args.zkp_clip,
                    "tau": args.zkp_tau,
                    "diff_bits": max(1, args.groth16_diff_bits),
                    "groth16_bin": args.groth16_bin,
                    "groth16_pk": args.groth16_pk,
                    "groth16_vk": args.groth16_vk,
                    "layer_whitelist_path": args.groth16_layer_whitelist,
                }
            )

    malicious_clients = {cid.strip() for cid in args.malicious_clients.split(",") if cid.strip()}
    malicious_rounds = {
        int(r.strip())
        for r in args.malicious_rounds.split(",")
        if r.strip()
    }
    malicious_config = {
        "clients": malicious_clients,
        "scale": float(args.malicious_scale),
        "fraction": float(args.malicious_fraction),
        "alpha": float(args.malicious_alpha),
        "clip": float(args.malicious_clip),
        "attack_prob": float(args.malicious_attack_prob),
        "mode": args.malicious_mode,
        "noise_scale": float(args.malicious_noise_scale),
        "rounds": malicious_rounds,
    }
    if args.malicious_layer_whitelist:
        malicious_config["layer_whitelist_path"] = args.malicious_layer_whitelist

    client_fn = client_fn_simulation(
        partitioner=partitioner,
        valloader=valloader,
        class_weights=class_weights,
        zkp_config=zkp_config,
        malicious_config=malicious_config,
    )

    # 4. 获取服务器组件 (Strategy 和 ServerConfig)
    #    在这里设置您想要的训练轮数
    server_components = get_server_components(
        num_rounds=args.num_rounds,
        num_clients=NUM_CLIENTS,
        eval_fraction=args.eval_fraction,
        fit_fraction=args.fit_fraction,
        local_epochs=args.local_epochs,
        zkp_config=zkp_config,
    )

    # 5. 定义客户端所需的计算资源
    client_resources = {"num_cpus": 2, "num_gpus": 0.5}

    # 6. 启动联邦学习仿真
    print("--- Starting Flower Simulation ---")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=server_components.config,
        strategy=server_components.strategy,
        client_resources=client_resources,
    )

    print("\n--- Simulation Finished ---")
    chain_metrics = getattr(server_components.strategy, "round_chain_metrics", None)
    print_history_summary(
        history,
        total_rounds=args.num_rounds,
        num_clients=NUM_CLIENTS,
        chain_metrics=chain_metrics,
    )

    # 7. 关闭 Ray
    ray.shutdown()
