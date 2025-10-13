from typing import List, Dict
from .node import Node

class Network:
    """
    一个简单的P2P网络模拟器。
    """
    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def join(self, node: Node):
        """一个新节点加入网络"""
        print(f"网络: 节点 {node.node_id} 已加入。")
        self.nodes[node.node_id] = node

    def broadcast(self, sender_id: str, message: str):
        """
        将消息从一个发送者广播给网络中的所有其他节点。
        """
        for node_id, node in self.nodes.items():
            if node_id != sender_id:
                # 在真实网络中，这将通过网络套接字发送
                # 这里我们直接调用接收方法来模拟
                node.receive_message(sender_id, message)
