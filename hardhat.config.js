// hardhat.config.js

// --- [核心修复] ---
// 不再导入整个 toolbox，只导入我们需要的、已安装的核心插件
require("@nomicfoundation/hardhat-ethers");

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: "0.8.20",
  networks: {
    hardhat: {},
    ganache: {
      url: "http://127.0.0.1:8545",
    },
  },
};