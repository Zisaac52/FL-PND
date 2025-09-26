// scripts/deploy.js
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("Fetching contract factory for 'FederatedLearning'...");
  
  const FederatedLearningFactory = await hre.ethers.getContractFactory("FederatedLearning");
  
  console.log("Deploying contract...");
  const federatedLearning = await FederatedLearningFactory.deploy();

  await federatedLearning.waitForDeployment();

  const contractAddress = await federatedLearning.getAddress();
  console.log(`FederatedLearning contract deployed to: ${contractAddress}`);
  
  saveFrontendFiles(contractAddress);
}

function saveFrontendFiles(contractAddress) {
  const contractsDir = path.join(__dirname, "..", "fl-pnd", "fl_pnd", "contracts");

  if (!fs.existsSync(contractsDir)) {
    fs.mkdirSync(contractsDir, { recursive: true });
  }

  // 保存合约地址
  fs.writeFileSync(
    path.join(contractsDir, "contract-address.json"),
    JSON.stringify({ address: contractAddress }, undefined, 2)
  );

  // 保存合约的 ABI
  const FederatedLearningArtifact = hre.artifacts.readArtifactSync("FederatedLearning");
  
  fs.writeFileSync(
    path.join(contractsDir, "FederatedLearning.json"),
    JSON.stringify({ abi: FederatedLearningArtifact.abi }, null, 2)
  );
  
  console.log("Contract address and ABI saved to fl-pnd/fl_pnd/contracts/");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});