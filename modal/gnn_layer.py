import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_similarity(input):
    #cos_matrix=torch.zeros(input.size(0),input.size(1),input.size(1))
    list=[]
    for i in range(input.size(0)):
        cos_sim=F.cosine_similarity(input[i].unsqueeze(1),input[i].unsqueeze(0),dim=-1)
        list.append(cos_sim)
    cos_matrix=torch.stack(list,dim=0)
    return cos_matrix

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features,alpha,negative_atten=True,dropout=0.1):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features))).cuda()
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1))).cuda()
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.wtrans = nn.Parameter(torch.zeros(size=(2 * in_features, out_features))).cuda()
        nn.init.xavier_uniform_(self.wtrans.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.negative_atten=negative_atten

    def forward(self, inp, adj): #input (batch,num,input_dim)
        h = torch.matmul(inp, self.W)#h (batch,num,hidden=512)
        Wh1 = torch.matmul(h, self.a[:self.out_features, :])#wh1 (batch,num,1)
        Wh2 = torch.matmul(h, self.a[self.out_features:, :])#wh2 (batch,num,1)
        e = Wh1 + Wh2.transpose(1,2) # e(batch,num,num)
        zero_vec = -1e12 * torch.ones_like(e)
        e = self.leakyrelu(e)
        attention = torch.where(adj > 0, e, zero_vec)
        negative_attention = torch.where(adj > 0, -e, zero_vec)
        # attention=self.leakyrelu(attention)
        # negative_attention =self.leakyrelu(negative_attention)
        attention = F.softmax(attention, dim=2)
        negative_attention = -F.softmax(negative_attention,dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        negative_attention = F.dropout(negative_attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, inp)#(batch,num,num) x (batch,num,input_dim)--> (batch,num,input_dim)
        h_prime_negative = torch.matmul(negative_attention, inp)
        h_prime_double = torch.cat([h_prime,h_prime_negative],dim=2)#(batch,num,input_dim*2)
        new_h_prime = torch.matmul(h_prime_double,self.wtrans)#(batch,num,hidden_dim)

        #res
        if self.negative_atten==True:
            return F.elu(new_h_prime)
        else:
            return F.elu(h_prime)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, residual=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        if not residual:
            self.residual = lambda x: 0
        elif (in_features == out_features):
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=5, padding=2)
    def reset_parameters(self):
        # stdv = 1. / sqrt(self.weight.size(1))
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.1)

    def forward(self, input, adj):
        # To support batch operations
        support = input.matmul(self.weight)
        output = adj.matmul(support)

        if self.bias is not None:
            output = output + self.bias
        if self.in_features != self.out_features and self.residual:
            input = input.permute(0,2,1)
            res = self.residual(input)
            res = res.permute(0,2,1)
            output = output + res
        else:
            output = output + self.residual(input)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
