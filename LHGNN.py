# 卷积层：节点 ——>超边 ———> 节点
# 一共是两层
# 在卷积层开始对节点特征进行batchnorm
# 在第一层最后的节点嵌入加dropout
# 激活函数使用的Relu

import torch
import torch.nn as nn
from dhg.structure.hypergraphs import Hypergraph
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LHGNNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        De,
        De_sum,
        Dv,
        Dv_sum,
        drop_rate: float = 0.5,
        is_last: bool = False
        # is_first: bool = False
    ):
        super().__init__()
        self.is_last = is_last
        # self.is_first = is_first
        self.bn = nn.BatchNorm1d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.linear_e = nn.Linear(in_channels, in_channels, bias=True)
        self.linear_v = nn.Linear(in_channels, out_channels, bias=True)

        self.De = De
        self.De_sum = De_sum
        self.Dv = Dv
        self.Dv_sum = Dv_sum
        self.atten_neighbor = nn.Linear(in_channels, 1, bias=True)
        self.atten_self = nn.Linear(in_channels, 1, bias=True)

    def forward(self, X_origin: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        # 做batchNorm
        X = self.bn(X_origin)

        # 节点消息传递到超边
        X = torch.sparse.mm(self.Dv, X)
        Y = torch.sparse.mm(hg.H_T, X)
        Y = torch.sparse.mm(self.De_sum, Y)

        # 超边特征过线性层转换维度
        # Y = self.linear_e(Y)
        Y = self.act(Y)

        # 超边消息传递到节点
        Y = torch.sparse.mm(self.De, Y)
        X = torch.sparse.mm(hg.H, Y)
        X = torch.sparse.mm(self.Dv_sum, X)

        # # 节点特征过线性层转换维度
        X = self.linear_v(X)
        # X = self.act(X)

        if not self.is_last:
            X = self.drop(X)
        return X

class LHGNN(nn.Module):
    def __init__(self,
        in_channels: int,
        hid_channels: int,
        output_channels: int,
        De,
        De_sum,
        Dv,
        Dv_sum,
        drop_rate: float = 0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            LHGNNConv(in_channels, hid_channels,De=De, De_sum=De_sum, Dv=Dv, Dv_sum=Dv_sum, drop_rate=drop_rate)
        )
        # for i in range(0,6):
        #     self.layers.append(
        #         PaperHNHNConv(hid_channels, hid_channels, De=De, De_sum=De_sum, Dv=Dv, Dv_sum=Dv_sum, drop_rate=drop_rate)
        #     )
        self.layers.append(
            LHGNNConv(hid_channels, output_channels,De=De, De_sum=De_sum, Dv=Dv, Dv_sum=Dv_sum, drop_rate=drop_rate, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        for layer in self.layers:
            X = layer(X, hg)
        return X