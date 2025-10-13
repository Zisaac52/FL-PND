# Ladder-FL: 基于双链DAG的去中心化联邦学习原型

## 1. 项目概述

`Ladder-FL` 是一个基于《Ladder: A Reliable Leaderless Blockchain for Decentralized Federated Learning》论文核心思想实现的原型项目。它旨在探索如何通过一种去中心化的方式，利用创新的**双链DAG（有向无环图）**结构取代传统的中心化服务器，从而实现一个更健壮、更可靠、无单点故障的联邦学习系统。

传统的联邦学习严重依赖中心服务器进行模型分发和聚合，这使得服务器成为性能瓶颈和潜在的攻击目标。本项目通过模拟一个点对点网络，展示了客户端如何协同工作，在没有领导者的情况下完成模型的聚合与同步。

## 2. 如何运行

请确保您已正确安装 Conda 并创建了项目所需的环境。

**步骤 1: 激活 Conda 环境**

```bash
conda activate fl_project
```

**步骤 2: 运行联邦学习模拟**

执行以下命令以启动整个模拟流程。脚本将初始化多个客户端节点，并开始进行多轮的联邦学习。

```bash
python run_federated.py
```

您将在终端看到每一轮的训练、聚合和模型评估日志。

## 3. 核心架构

本项目的核心是**双链DAG模型**，它由两种不同类型的区块组成，共同维护着联邦学习的全局状态和历史记录。

-   **模型链 (Upper-Chain)**: 这是一条记录各个客户端**本地模型更新**的链。每个客户端在本地完成一轮训练后，都会生成一个`UpperChainBlock`。这个区块不仅包含训练后的模型参数，还指向上一个它所基于的全局模型区块（`LowerChainBlock`），从而形成一个有向无环图（DAG）。这允许多个客户端并行地进行训练和提交，而无需等待中心服务器的协调。

-   **聚合链 (Lower-Chain)**: 这是一条记录**全局模型聚合**的链。网络中的节点（在本项目中由一个“收敛节点”角色模拟）会收集来自不同客户端的`UpperChainBlock`，并根据其中记录的数据集大小对模型参数进行加权平均。聚合完成后，会生成一个包含新全局模型的`LowerChainBlock`并广播到全网。这条链构成了系统的主干，为所有客户端提供了同步和共识的基准。

这种双链结构巧妙地将“本地训练”和“全局聚合”两个过程解耦，通过密码学哈希指针将它们安全地连接起来，从而实现了去中心化的联邦学习流程。

## 4. 区块结构

为了实现上述架构，我们定义了两种核心数据结构：`UpperChainBlock` 和 `LowerChainBlock`。

### UpperChainBlock

由客户端在本地训练后生成，记录了一次独立的模型更新。

-   `client_id` (str): 产生该区块的客户端的唯一标识符。
-   `dataset_size` (int): 客户端用于本轮训练的数据样本数量。这个值在模型聚合时用作加权平均的权重。
-   `model_params` (ModelParameters): 客户端在本地数据上训练后得到的模型参数。
-   `parent_upper_hash` (str): 指向上一个由**同一客户端**生成的`UpperChainBlock`的哈希。这形成了一条记录该客户端模型演进历史的子链。
-   `parent_lower_hash` (str): 指向客户端本轮训练所基于的`LowerChainBlock`的哈希。这明确了本次本地更新是在哪个全局模型版本的基础上进行的。
-   `metrics` (Dict[str, float]): 包含本轮训练的性能指标，例如损失（loss）或准确率（accuracy）。
-   `nonce` (int): 工作量证明中使用的随机数。在当前简化版本中，此字段保留但未使用。
-   `hash` (str): 该区块自身的SHA-256哈希值，由区块内的核心内容计算得出，用于保证数据的完整性和不可篡改性。

### LowerChainBlock

由收敛节点在聚合多个`UpperChainBlock`后生成，代表了一轮全局模型的共识。

-   `round` (int): 当前联邦学习的轮次编号。
-   `standard_upper_block_hash` (str): 在聚合时，被选为“标准”或“主干”的`UpperChainBlock`的哈希。
-   `forked_upper_block_hashes` (List[str]): 其他参与本轮聚合的并行`UpperChainBlock`的哈希列表。
-   `aggregated_global_model` (ModelParameters): 通过对多个`UpperChainBlock`中的模型进行加权平均后得到的新一轮全局模型参数。
-   `parent_lower_hash` (str): 指向上一个`LowerChainBlock`的哈希，形成了全局模型的演进主链。
-   `hash` (str): 该区块自身的SHA-256哈希值，确保全局模型状态的完整性。

## 5. 详细工作流程

一轮完整的去中心化联邦学习流程如下：

1.  **获取全局模型**: 客户端（如 Client A）首先从网络中同步最新的`LowerChainBlock`，并提取其中的`aggregated_global_model`作为本轮训练的初始模型。
2.  **本地训练**: Client A 使用自己的本地数据集对模型进行训练（例如，一个或多个epoch）。
3.  **生成上链区块**: 训练完成后，Client A 会创建一个`UpperChainBlock`。该区块包含了更新后的模型参数、本地数据集的大小、以及指向父区块（上一个`LowerChainBlock`和它自己的上一个`UpperChainBlock`）的哈希。
4.  **广播与并行**: Client A 将新生成的`UpperChainBlock`广播到网络中。与此同时，其他客户端（Client B, Client C 等）也在独立地重复步骤1-3，从而在网络中形成了多个并行的`UpperChainBlock`。
5.  **收敛与聚合**: 网络中的某个节点（“收敛节点”）监听并收集这些基于同一个`LowerChainBlock`生成的`UpperChainBlock`。
6.  **加权平均**: 当收集到足够数量的区块后，收敛节点会读取每个`UpperChainBlock`中的`dataset_size`和`model_params`，并执行加权平均算法，计算出新的全局模型。
7.  **生成下链区块**: 收敛节点将新的全局模型、当前轮次信息以及所有被聚合的`UpperChainBlock`的引用打包成一个新的`LowerChainBlock`。
8.  **广播新一轮全局模型**: 收敛节点将这个新的`LowerChainBlock`广播到全网。
9.  **开始新一轮**: 所有客户端接收到这个新的`LowerChainBlock`后，便以此为基础，开始下一轮的联邦学习，循环返回步骤1。

## 6. 流程图

以下时序图直观地展示了上述工作流程中节点间的交互：

```mermaid
sequenceDiagram
    participant Client A
    participant Client B
    participant Client C
    participant Convergence Node
    participant Network

    Note over Client A, Client C: Start of Round N
    Client A->>Network: Get Latest LowerChainBlock (Global Model N)
    Client B->>Network: Get Latest LowerChainBlock (Global Model N)
    Client C->>Network: Get Latest LowerChainBlock (Global Model N)

    par Local Training
        Client A->>Client A: Train on local data
        Client B->>Client B: Train on local data
        Client C->>Client C: Train on local data
    end

    Client A->>Network: Broadcast UpperChainBlock A
    Client B->>Network: Broadcast UpperChainBlock B
    Client C->>Network: Broadcast UpperChainBlock C

    Note over Convergence Node: Collects blocks for Round N
    Convergence Node->>Network: Receive UpperChainBlock A, B, C

    Convergence Node->>Convergence Node: Aggregate models (weighted average) to create Global Model N+1

    Convergence Node->>Network: Broadcast new LowerChainBlock (contains Global Model N+1)

    Note over Client A, Client C: Start of Round N+1
    Client A->>Network: Get Latest Lower-Chain Block (Global Model N+1)
    Client B->>Network: Get Latest Lower-Chain Block (Global Model N+1)
    Client C->>Network: Get Latest Lower-Chain Block (Global Model N+1)