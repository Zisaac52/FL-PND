# fl_pnd/client_app.py
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import flwr as fl
import torch
import hashlib
from web3 import Web3
from flwr.common import Context
from .dataset import get_dataloader
from .task import get_net, set_parameters, train, test

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, partitioner, valloader, class_weights, contract_address, contract_abi):
        self.cid = cid
        self.net = get_net().to(DEVICE)
        self.class_weights = class_weights
        
        # --- [区块链设置] ---
        self.w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
        self.contract = self.w3.eth.contract(address=contract_address, abi=contract_abi)
        # 为每个客户端分配一个不同的 Ganache 账户 (0号留给服务器)
        # 注意：在真实的去中心化系统中，每个客户端将管理自己的私钥
        account_index = (int(self.cid) % 9) + 1 # 确保索引在 1-9 之间
        self.account = self.w3.eth.accounts[account_index]
        self.w3.eth.default_account = self.account
        
        # --- [数据加载] ---
        partition = partitioner.load_partition(int(cid))
        print(f"客户端 {self.cid} (地址: {self.account}) 已创建，加载了 {len(partition)} 个训练样本。")
        self.trainloader = get_dataloader(partition, batch_size=4, is_train=True)
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        current_round = config["server_round"]
        
        train(
            net=self.net, 
            trainloader=self.trainloader, 
            epochs=1, 
            device=DEVICE, 
            class_weights=self.class_weights,
            current_round=current_round
        )
        
        local_params = self.get_parameters(config={})
        
        # --- [区块链交互] ---
        params_bytes = b"".join([param.tobytes() for param in local_params])
        update_hash = hashlib.sha256(params_bytes).digest() # .digest() 返回 bytes
        
        print(f"客户端 {self.cid} 正在向区块链提交哈希...")
        try:
            tx_hash = self.contract.functions.submitUpdate(current_round, update_hash).transact()
            self.w3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"客户端 {self.cid} 提交哈希成功！Tx: {tx_hash.hex()}")
        except Exception as e:
            print(f"客户端 {self.cid} 提交哈希失败: {e}")
            
        return local_params, len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, metrics_dict = test(net=self.net, testloader=self.valloader, device=DEVICE)
        return float(loss), len(self.valloader.dataset), metrics_dict

def client_fn_simulation(partitioner, valloader, class_weights, contract_address, contract_abi):
    def client_fn(context: Context) -> fl.client.Client:
        cid = str(context.node_config.get("partition-id", context.node_id))
        return FlowerClient(
            cid, partitioner, valloader, class_weights, contract_address, contract_abi
        ).to_client()
    return client_fn