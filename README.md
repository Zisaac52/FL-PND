# Federated Learning for Panax Notoginseng Disease Segmentation

This project implements a federated learning system for segmenting diseases on Panax Notoginseng (三七) leaves. It is built using the [Flower](https://flower.ai/) framework and achieves high performance by fine-tuning a **U-Net** model with a pre-trained ResNet-50 backbone.

The system is designed to handle the significant class imbalance present in the dataset (large background areas vs. small disease regions) and uses a robust evaluation setup with a global validation set.

  
## Key Features & Optimizations

- **High-Performance U-Net Architecture**: Employs a U-Net model, whose skip-connections are proven to be highly effective for tasks requiring precise localization and fine-grained boundary segmentation.
- **Federated Averaging (FedAvg)**: Utilizes the foundational federated learning algorithm to collaboratively train a global model without sharing sensitive raw data.
- **Two-Stage Fine-Tuning**: Implements a professional transfer learning strategy:
  1.  **Stage 1**: The model's pre-trained encoder is frozen, and only the decoder is trained to rapidly adapt to the new task.
  2.  **Stage 2**: The entire network is unfrozen and fine-tuned end-to-end with a very small learning rate for maximum performance.
- **Class Imbalance Handling**:
  - **Weighted Cross-Entropy Loss**: Applies pre-calculated class weights to counteract the dominance of the background class.
  - **Foreground-Focused Metrics**: Both the loss function and evaluation metrics (mIoU, Pixel Accuracy) ignore the background class to provide a true measure of disease segmentation performance.
- **GPU Accelerated Simulation**: Launched via a custom Python script that provides full control over the Ray backend, ensuring efficient GPU utilization.

## Project Structure

```
PND/
├── Panax notoginseng disease dataset/    # Dataset (not included in repo)
│   └── VOC2007/
│       ├── JPEGImages/
│       ├── SegmentationClass/
│       └── ImageSets/
├── fl-pnd/                               # The Flower project package
│   ├── fl_pnd/
│   │   ├── client_app.py                 # Defines the Flower client logic
│   │   ├── dataset.py                    # Custom dataset loading and utilities
│   │   ├── server_app.py                 # Defines the Flower server logic
│   │   └── task.py                       # Defines the ML model, train/test functions
│   └── pyproject.toml                    # Project dependencies and metadata
└── run.py                                # Main script to launch the simulation
```

## Setup and Installation

### Prerequisites

- Python 3.9+
- An NVIDIA GPU with CUDA installed
- `git` for cloning the repository

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Zisaac52/FL-PND.git
    cd FL-PND
    ```

2.  **Download the Dataset:**
    Download the "Panax notoginseng disease dataset" and place it in the project's root directory, ensuring the path is `PND/Panax notoginseng disease dataset/`. This repository does not include the dataset itself.

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install dependencies:**
    The project and its dependencies are defined in `fl-pnd/pyproject.toml`. Install the project in editable mode:
    ```bash
    pip install -e fl-pnd/
    ```

## How to Run

The simulation is launched using the custom `run.py` script, which now orchestrates Ray setup, Flower sampling, and the Ladder-inspired reputation chain.

Simply run the following command from the project root directory (`PND/`):

```bash
python run.py
```

The script will:
1.  Initialize a local Ray cluster with GPU support.
2.  Load the dataset and calculate class weights.
3.  Start the Ladder-aware Flower simulation for the number of rounds you specify (either in the CLI or `fl-pnd/server_app.py` defaults).
4.  Print the final training history (loss and foreground mIoU) plus a compact per-round summary table.

### Customize the Simulation

All core hyperparameters can be passed directly to `run.py`:

```bash
python run.py \
    --num-clients 10 \
    --num-rounds 5 \
    --local-epochs 2 \
    --fit-fraction 0.8 \
    --eval-fraction 0.3
```

- `--num-clients`: total virtual clients managed by Ray (default `10`)
- `--num-rounds`: number of federated rounds (default `3`)
- `--local-epochs`: local epochs per selected client (default `1`)
- `--fit-fraction`: fraction of clients sampled for training (default `1.0`)
- `--eval-fraction`: portion of clients sampled for evaluation (default `0.3`)

These flags flow directly into the custom Flower strategy, so both training and evaluation automatically obey the configuration you pass at runtime.

### Payload Compression & Lighter Clients

Clients now transmit **float16 model deltas (Δw)** instead of full precision weights. For each round, the client caches the received global model `W_t`, trains locally to obtain `W'_t`, computes `Δw = W'_t - W_t`, and uploads the FP16-compressed delta. The server aggregates the deltas, reconstructs the next global model, and keeps full weights inside the LowerChainBlock.  
同时，`fit()` 阶段不再执行昂贵的验证集评估，而是直接使用训练过程返回的平均损失来更新信誉。这两个改动显著降低了 Training Latency，并将 Upload (MB) 减少到原来的约一半，为后续的 Top-K/量化等进阶压缩留出了空间。

### Blockchain-Oriented Metrics

`run.py` now reports a second table after each simulation which captures the DAG/blockchain perspective:

| Metric | Description |
| --- | --- |
| `Latency (s)` | Wall-clock time spent in each federated round (from sampling to LowerChainBlock creation). |
| `Training Latency (s)` | Max local training time among all clients in the round. |
| `Consensus Latency (s)` | Portion of the round spent after training (aggregation, block creation). |
| `Throughput (blocks/s)` | Confirmed UpperChainBlocks per second (standard + forks). |
| `Upload (MB)` | Aggregate model bytes uploaded to the server that round. |
| `Blocks` / `Forks` | Count of standard + forked blocks included in the LowerChainBlock. |

These stats are logged automatically and can be copy/pasted into experiment reports for latency/吞吐量分析。

## Future Work

The next major step for this project is to integrate blockchain technology to enhance the security, traceability, and incentive mechanisms of the federated learning process.
