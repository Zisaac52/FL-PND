# 基于Ladder-DAG的去中心化联邦学习架构设计

## 1. 概述

本文档旨在为现有中心化联邦学习项目设计一个全新的、基于 `Ladder` 论文提出的结构化DAG区块链的去中心化架构。新架构将取代原有的中心化服务器，以解决单点故障、信任问题，并为未来引入零知识证明 (ZKP) 等隐私保护技术奠定基础。

## 2. 核心概念：Ladder联邦学习 (LadderFL)

我们将 `Ladder` 的双链DAG模型应用于联邦学习场景，定义如下：

*   **Upper-Chain (模型链)**: 一条由联邦学习客户端（节点）并行生成的DAG链。每个区块代表一次本地模型训练的结果。
*   **Lower-Chain (聚合链)**: 一条线性的、负责收敛和聚合模型链的区块链。每个区块代表一次全局模型的聚合与更新，并包含最新的全局模型。

这种结构将中心化服务器的**模型分发**和**模型聚合**功能，解构到了一个去中心化的、由所有节点共同维护的协议中。

## 3. 架构详述

### 3.1 区块定义

#### 3.1.1 Upper-Chain Block (上链区块 / 模型更新区块)

由每个客户端在完成本地训练后创建和广播。

*   **内容**:
    *   `model_params`: 客户端本地训练后的模型参数。
    *   `client_id`: 客户端的唯一标识。
    *   `dataset_size`: 用于本地训练的数据样本数量（用于FedAvg加权）。
    *   `parent_upper_hash`: 指向上一个标准上链区块的哈希。
    *   `parent_lower_hash`: 指向上一个下链区块的哈希。
    *   `zkp_proof` (可选): 未来用于验证训练过程的零知识证明。
*   **生成方式**: 客户端需要解决一个低难度的 **Proof of Work (PoW)**。这主要是为了防止女巫攻击和网络垃圾信息，而非共识机制。

#### 3.1.2 Lower-Chain Block (下链区块 / 全局模型区块)

由每一轮被选出的 **收敛节点 (Convergence Node)** 创建，是整个系统的“心跳”。

*   **内容**:
    *   `round`: 当前的训练轮次。
    *   `standard_upper_block_hash`: 根据“最难链原则”从本轮所有上链区块中选出的**标准区块**的哈希。
    *   `forked_upper_block_hashes`: 本轮所有其他有效（但未被选为标准）的上链区块的哈希列表。
    *   `aggregated_global_model`: **核心字段**。由收敛节点执行FedAvg算法后计算出的、最新的全局模型参数。
    *   `parent_lower_hash`: 指向上一个下链区块的哈希。
*   **生成方式**: 无需PoW。由上一轮 `standard_upper_block` 的创建者（即上一轮的“优胜者”）担任本轮的收敛节点，负责生成并广播。

### 3.2 核心问题解答

#### 3.2.1 角色替换：DAG网络如何取代中心化服务器？

*   **模型分发**: 客户端通过监听和获取最新的 `Lower-Chain Block` 来获得全局模型，取代了服务器的“分发”功能。
*   **模型聚合**: 由每一轮的**收敛节点**负责执行聚合算法，取代了服务器的“聚合”功能。这是一个动态的、轮换的角色，避免了中心化。
*   **训练协调**: `Lower-Chain Block` 的产生标志着一轮训练的结束和新一轮的开始，其 `round` 字段同步了整个网络的训练节奏。

#### 3.2.2 模型分发与聚合流程

1.  **分发**: 客户端 `C` 监听到新的 `Lower-Chain Block (Round r)` 被网络接受。
2.  `C` 从该区块中解析出 `aggregated_global_model` 字段，并将其加载为本地模型。
3.  **本地训练**: `C` 在本地数据集上进行训练。
4.  **发布**: 训练完成后，`C` 创建一个 `Upper-Chain Block`，引用 `Lower-Chain Block (Round r)` 的哈希，并通过PoW挖矿后广播出去。
5.  **聚合 (由收敛节点 `S` 执行)**:
    *   节点 `S` (上一轮的标准区块创建者) 收集所有指向 `Lower-Chain Block (Round r)` 的 `Upper-Chain Block`。
    *   等待一个预设的超时时间后，`S` 根据“最难链原则”选出本轮的 `standard_upper_block`。
    *   `S` 下载所有被引用的上链区块（标准+分叉）中的 `model_params` 和 `dataset_size`。
    *   `S` 在本地执行联邦平均算法 (FedAvg)。
    *   `S` 创建并广播包含新全局模型的 `Lower-Chain Block (Round r+1)`。

#### 3.2.3 完整工作流程 (Mermaid图)

```mermaid
graph TD
    A[Start: Client Joins Network] --> B{Sync DAG};
    B --> C[Find Latest Lower-Chain Block];
    C --> D[Load Global Model];
    D --> E[Local Training];
    E --> F[Create Upper-Chain Block (with PoW)];
    F --> G[Broadcast to Network];
    G --> H{Wait for next Lower-Chain Block};
    
    subgraph Convergence Node (Round r+1)
        I[Collect Upper-Chain Blocks for Round r] --> J[Select Standard Block via Hardest Chain Principle];
        J --> K[Download All Model Updates];
        K --> L[Perform FedAvg Aggregation];
        L --> M[Create & Broadcast Lower-Chain Block (Round r+1)];
    end

    H -- Receives New Block --> C;
    M -- Broadcasts to all clients --> H;

```

#### 3.2.4 验证与共识

*   **模型更新验证**:
    *   **收敛节点初步验证**: 收敛节点在聚合前，可以对每个上链区块的模型进行快速抽样测试。如果发现恶意更新（如模型投毒导致性能骤降），可以拒绝将其纳入聚合，并在下链区块中进行标记，作为对该节点的惩罚依据。
    *   **社区交叉验证**: 所有节点都可以验证下链区块中的聚合结果是否正确。

*   **网络共识与安全**:
    *   **最难链原则 (Hardest Chain Principle)**: 确保了标准区块和收敛节点的选择是公平且基于贡献的，有效抵抗自私挖矿和活性攻击。
    *   **BFT委员会与Super Block**: 当收敛节点作恶或超时，系统将激活由最近N个标准区块创建者组成的BFT委员会。委员会通过 **HotStuff** 等BFT共识算法，生成一个 `Super Block` 来取代错误的下链区块，从而保证网络的安全和活性。

## 4. 未来扩展：集成零知识证明 (ZKP)

ZKP可以无缝集成到此架构中，以增强隐私和安全性。

*   **流程**: 客户端在创建 `Upper-Chain Block` 时，额外生成一个zk-SNARK证明，并将其放入 `zkp_proof` 字段。
*   **证明内容**:
    1.  训练的**起点**是正确的全局模型。
    2.  训练过程遵循了预设的**算法和超参数**。
    3.  训练数据满足某些**统计特性**（例如，数据量在某个范围内）。
*   **优势**: 收敛节点不再需要信任客户端，只需验证一个轻量的证明，即可确信模型更新的有效性和诚实性，极大地提高了系统的抗攻击能力。

## 5. 结论

基于Ladder-DAG的去中心化联邦学习架构 (LadderFL) 成功地将中心化服务器的功能解构为一个健壮、高效、安全的去中心化协议。它不仅解决了单点问题，还通过其结构化的设计为未来的隐私增强技术（如ZKP）提供了完美的集成点。这份设计文档将作为后续编码实现的指导蓝图。