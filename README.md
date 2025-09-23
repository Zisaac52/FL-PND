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

The simulation is launched using the custom `run.py` script, which provides full control over the Ray backend and resource allocation.

Simply run the following command from the project root directory (`PND/`):

```bash
python run.py
```

The script will:
1.  Initialize a local Ray cluster with GPU support.
2.  Load the dataset and calculate class weights.
3.  Start the Flower simulation for the number of rounds specified in `fl-pnd/server_app.py`.
4.  Print the final training history (loss and foreground mIoU) upon completion.

## Future Work

The next major step for this project is to integrate blockchain technology to enhance the security, traceability, and incentive mechanisms of the federated learning process.