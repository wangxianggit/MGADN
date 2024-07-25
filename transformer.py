import torch
from torch import nn


# class Decoder(nn.Module):
#     def __init__(self, in_feats,h_feats,n_head,dropout_rate,n_layers):
#         super().__init__()

#         self.layers = nn.ModuleList([DecoderLayer(in_feats=in_feats,h_feats=h_feats,n_head=n_head,
#                                                   dropout_rate=0.)for _ in range(n_layers)])

#         self.act_fn = nn.ReLU()
#         self.lin = nn.Linear(h_feats,in_feats)
#         self.mlp = nn.Sequential(nn.Linear(in_feats,h_feats) )

#     def forward(self, x, edge_index):
#         _x = x
#         for layer in self.layers:
#             x = layer(x, edge_index)
#             x = x + _x
#             # x = self.lin(x)
#         output = self.mlp(x)
#         return output

class Decoder(nn.Module):
    def __init__(self, in_feats,h_feats,n_head,dropout_rate,n_layers):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(in_feats=in_feats,h_feats=h_feats,n_head=n_head,
                                                  dropout_rate=0.)for _ in range(n_layers)])
        self.act_fn = nn.ReLU()
        self.mlp = nn.Sequential(nn.Linear(in_feats,h_feats) )

    def forward(self, x, edge_index):
        _x = x
        for layer in self.layers:
            x = layer(x, edge_index)
            x = x + _x
          
        output = self.mlp(x)
        return output

class DecoderLayer(nn.Module):

    def __init__(self, in_feats, h_feats, n_head, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(in_channels=in_feats, hid_channels=h_feats, n_head=n_head)
        self.linear = nn.Linear(in_feats, h_feats)
        self.norm1 = LayerNorm(hid_channels= h_feats)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.norm3 = LayerNorm(hid_channels= h_feats)
        self.linear1 = nn.Linear(h_feats, in_feats)

    def forward(self, x, edge_index):
    
        _x = x
        x = self.self_attention(q=x, k=x, v=x)
        x = self.dropout1(x)
        _x = self.linear(_x)
      
        x = self.norm1(x + _x)
      
        x = self.linear1(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, hid_channels, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hid_channels))
        self.beta = nn.Parameter(torch.zeros(hid_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
   

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, in_channels, hid_channels,n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(in_channels, hid_channels)
        self.w_k = nn.Linear(in_channels, hid_channels)
        self.w_v = nn.Linear(in_channels, hid_channels)
        self.w_concat = nn.Linear(hid_channels, hid_channels)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, tensor):
        length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        length , head, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(length, d_model)
        return tensor

class PositionwiseFeedForward(nn.Module):

    def __init__(self, in_channels, hid_channels, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(hid_channels, hid_channels)
        self.linear2 = nn.Linear(hid_channels, in_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


import math
from torch import nn

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        head, length, d_tensor = k.size()
        k_t = k.transpose(1, 2)  
        score = (q @ k_t) / math.sqrt(d_tensor)  
        score = self.softmax(score)
        v = score @ v

        return v, score