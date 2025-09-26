// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// 引入 Hardhat 的 console.log 功能，方便调试
import "hardhat/console.sol";

contract FederatedLearning {
    address public server; // 聚合服务器的地址
    uint256 public currentRound;

    struct ModelUpdate {
        address client;
        bytes32 updateHash; // 客户端提交的模型更新哈希
    }

    // 记录每一轮所有客户端提交的更新
    mapping(uint256 => ModelUpdate[]) public roundUpdates;
    // 记录每一轮最终的全局模型哈希
    mapping(uint256 => bytes32) public globalModelHashes;

    event UpdateSubmitted(uint256 indexed round, address indexed client, bytes32 updateHash);
    event RoundFinalized(uint256 indexed round, bytes32 aggregatedHash);

    constructor() {
        server = msg.sender; // 部署合约的地址就是服务器地址
        currentRound = 1;
        console.log("FederatedLearning contract deployed by:", msg.sender);
    }

    modifier onlyServer() {
        require(msg.sender == server, "Only the server can call this function");
        _;
    }

    // 客户端调用此函数提交更新凭证
    function submitUpdate(uint256 round, bytes32 updateHash) external {
        require(round == currentRound, "This round is not active");
        // 可以在这里增加检查客户端是否已注册的逻辑
        roundUpdates[round].push(ModelUpdate(msg.sender, updateHash));
        emit UpdateSubmitted(round, msg.sender, updateHash);
    }

    // 服务器在完成链下聚合后，调用此函数来敲定一轮
    function finalizeRound(bytes32 aggregatedHash) external onlyServer {
        globalModelHashes[currentRound] = aggregatedHash;
        emit RoundFinalized(currentRound, aggregatedHash);
        currentRound++; // 自动进入下一轮
    }

    // [可选] 添加一个视图函数，方便检查
    function getRoundUpdates(uint256 round) external view returns (ModelUpdate[] memory) {
        return roundUpdates[round];
    }
}