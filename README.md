# A Blockchain-based Federated Learning System for Panax Notoginseng Disease Segmentation

This project implements a blockchain-based federated learning system for segmenting diseases on Panax Notoginseng (三七) leaves. It is built using the [Flower](https://flower.ai/) framework and achieves high performance by fine-tuning a **U-Net** model with a pre-trained ResNet-50 backbone. The blockchain layer, built with **Solidity** and the **Hardhat framework**, provides a transparent and immutable ledger for auditing the training process, laying the groundwork for future decentralized trust and incentive mechanisms.

The system is designed to handle the significant class imbalance present in the dataset (large background areas vs. small disease regions) and uses a robust evaluation setup with a global validation set.

## Core Architecture & Workflow

The system adopts a **"Chain-off, Chain-on"** hybrid architecture to balance performance and decentralization:
-   **Off-chain (链下)**: Computationally intensive tasks like ML model training (U-Net on PyTorch) and aggregation (FedAvg) are performed on powerful servers to ensure efficiency.
-   **On-chain (链上)**: Trust-critical operations like recording contributions and finalizing rounds are executed on an Ethereum-compatible blockchain via a smart contract, ensuring transparency and auditability.

A typical training round proceeds as follows:
1.  **Task Initialization**: The Flower Server initiates a new training round.
2.  **Local Training (Off-chain)**: Clients train the global U-Net model on their private local data.
3.  **Hash Submission (On-chain)**: Each client submits a hash of its updated model to the Solidity smart contract, creating an immutable proof-of-contribution.
4.  **Parameter Upload (Off-chain)**: Clients send their full model parameters to the Flower Server.
5.  **Aggregation (Off-chain)**: The Flower Server aggregates the parameters using the FedAvg algorithm.
6.  **Round Finalization (On-chain)**: The server submits the hash of the new global model to the smart contract to conclude the round.

## Key Technical Features

### Machine Learning Core
- **Model**: **U-Net** with a pre-trained ResNet-50 backbone from the `segmentation-models-pytorch` library, proven effective for precise boundary detection.
- **Federated Learning Framework**: **Flower** (`flwr`), orchestrating the communication between the server and distributed clients.
- **Training Strategy**: A **two-stage fine-tuning** approach that first trains the decoder and then fine-tunes the entire network.
- **Optimizations**: **Weighted Cross-Entropy Loss** and **foreground-focused metrics** (mIoU, Pixel Accuracy) to handle severe class imbalance.

### Blockchain Layer
- **Smart Contract Language**: **Solidity** (version 0.8.20).
- **Development Framework**: **Hardhat**, a professional Ethereum development environment for compiling, deploying, testing, and debugging.
- **Local Blockchain**: The system is developed and tested against a local **Hardhat Network** node, providing an instant and flexible testing environment.


## Project Structure

```
PND/
├── contracts/                            # Solidity smart contracts
│   └── FederatedLearning.sol
├── fl-pnd/                               # The Flower Python project package
│   ├── fl_pnd/
│   │   ├── client_app.py                 # Client logic (ML + Blockchain interaction)
│   │   ├── contracts/                    # Deployed contract ABI and address (auto-generated)
│   │   ├── dataset.py
│   │   ├── server_app.py                 # Server strategy definition
│   │   └── task.py                       # U-Net model and train/test functions
│   └── pyproject.toml
├── scripts/                              # Deployment scripts for Hardhat
│   └── deploy.js
├── run.py                                # Main script to launch the entire system
├── hardhat.config.js                     # Hardhat configuration
├── package.json                          # NodeJS dependencies
└── .gitignore                            # Files to be ignored by Git
```

## Setup and Run Instructions

### Prerequisites

- Python 3.9+ & `pip`
- NVIDIA GPU with CUDA
- **Node.js** (v18+) & `npm`
- `git`


### Step 1: Clone & Install Python Dependencies

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Zisaac52/FL-PND.git
    cd FL-PND
    ```
2.  **Download the Dataset:**
    Place the "Panax notoginseng disease dataset" in the project's root directory (`PND/`).

3.  **Create a Python virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -e fl-pnd/
    ```

### Step 2: Setup Blockchain Environment

1.  **Install Node.js dependencies:**
    From the project root directory (`PND/`), run:
    ```bash
    npm install
    ```

2.  **Start the Local Blockchain**:
    Open a **new, dedicated terminal window**, navigate to the `PND/` directory, and run:
    ```bash
    npx hardhat node
    ```
    This will start a local Hardhat Ethereum node, which is a powerful alternative to Ganache. Keep this terminal running in the background.

### Step 3: Deploy the Smart Contract

1.  **Run the deployment script**:
    Open a **third terminal window**, navigate to the `PND/` directory, and run:
    ```bash
    npx hardhat run scripts/deploy.js --network ganache
    ```
    This will compile, deploy the `FederatedLearning.sol` contract to your local Hardhat node, and automatically save the contract's address and ABI to `fl-pnd/fl_pnd/contracts/`.

### Step 4: Run the Federated Learning Simulation

1.  **Execute the main script**:
    Go back to the terminal where you installed the Python dependencies (the second one) and run:
    ```bash
    python run.py
    ```

## Future Work & Research Directions

This project establishes a robust foundation for exploring advanced concepts in decentralized AI. The planned future work includes:

1.  **Enhanced Scalability with DAGs**:
    -   **Problem**: Traditional blockchains have limited throughput (transactions per second), which can become a bottleneck as the number of clients and training rounds increases.
    -   **Proposed Solution**: Integrate a **Directed Acyclic Graph (DAG)** based ledger. DAG structures allow for parallel transaction processing, offering significantly higher scalability and lower transaction fees compared to linear blockchains, making it more suitable for high-frequency updates in a large-scale FL system.

2.  **Verifiable Computation with Zero-Knowledge Proofs (ZKP)**:
    -   **Problem**: The current system relies on trusting the central server to perform the aggregation correctly. A malicious server could manipulate the global model.
    -   **Proposed Solution**: Implement a **zk-SNARK** protocol. The server would be required to generate a succinct cryptographic proof that the aggregation was performed correctly according to the FedAvg algorithm. Clients could then efficiently verify this proof on-chain via the smart contract without needing to re-execute the entire computation. This would achieve a state of "verifiable federated learning," removing the need to trust the central aggregator.

These future steps aim to transform the current auditable system into a fully decentralized, scalable, and mathematically verifiable platform for collaborative AI.