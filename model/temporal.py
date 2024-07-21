import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from .gnn_layer import *


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def get_sim_adj(x, mask):
    soft = nn.Softmax(-1)
    adj = calculate_similarity(x)
    adj = F.threshold(adj, 0.7, 0)  # (b,len,len)
    # mask (b,len)
    mask_matrix = torch.zeros_like(adj)
    for i in range(mask.shape[0]):
        frame_len = torch.count_nonzero(mask[i])  # 有效的序列长度
        mask_matrix[i, :frame_len, :frame_len] = 1

    adj = torch.where(mask_matrix > 0, adj, -1e9)
    adj = soft(adj)
    return adj


def get_distance_adj(batch_size, max_seqlen, mask):
    soft = nn.Softmax(-1)
    arith = np.arange(max_seqlen).reshape(-1, 1)
    dist = pdist(arith, metric='cityblock').astype(np.float32)  # 得到距离数组
    dist = torch.from_numpy(squareform(dist)).to('cuda')  # 得到距离矩阵，对角线为0
    dist = torch.exp(-dist / torch.exp(torch.tensor(1.)))
    dist = torch.unsqueeze(dist, 0).repeat(batch_size, 1, 1).to('cuda')
    # mask (b,len)
    mask_matrix = torch.zeros_like(dist)
    for i in range(mask.shape[0]):
        frame_len = torch.count_nonzero(mask[i])  # 有效的序列长度
        mask_matrix[i, :frame_len, :frame_len] = 1

    dist = torch.where(mask_matrix > 0, dist, -1e9)
    dist = soft(dist)
    return dist


class Local_Block(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, act=nn.GELU(), dropout=0):
        super(Local_Block, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = x.transpose(1, 2)  # (B,len,fea)-->(B,fea,len)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x


class Global_Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Global_Block, self).__init__()
        self.gcn1 = GraphConvolution(in_features=in_dim, out_features=out_dim, residual=True)
        self.gcn2 = GraphConvolution(in_features=in_dim, out_features=out_dim, residual=True)
        self.gcn3 = GraphConvolution(in_features=in_dim, out_features=out_dim, residual=True)
        self.gcn4 = GraphConvolution(in_features=in_dim, out_features=out_dim, residual=True)
        self.linear = nn.Linear(2 * out_dim, in_dim)
        self.gelu = QuickGELU()

    def forward(self, x, mask):
        sim_adj = get_sim_adj(x, mask)
        dis_adj = get_distance_adj(batch_size=x.shape[0], max_seqlen=x.shape[1], mask=mask)
        x1_h = self.gelu(self.gcn1(x, sim_adj))
        x1 = self.gelu(self.gcn2(x1_h, sim_adj))
        x2_h = self.gelu(self.gcn3(x, dis_adj))
        x2 = self.gelu(self.gcn4(x2_h, dis_adj))
        x = torch.cat((x1, x2), dim=-1)
        x = self.linear(x)

        return x


class temporal_encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(temporal_encoder, self).__init__()
        self.local_block = Local_Block(in_dim, hidden_dim, out_dim)
        self.global_block = Global_Block(in_dim, out_dim)

    def forward(self, x, mask):
        x = self.local_block(x)
        x = self.global_block(x, mask)
        return x
