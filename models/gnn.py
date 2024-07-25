import torch
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn.pytorch.conv as dglnn
from torch import nn
from transformer import Decoder
from dgl.nn.pytorch.conv import EdgeConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats=64, num_classes=2, num_layers=2, mlp_layers=1, dropout_rate=0.,
                 activation='ReLU', **kwargs):
        super().__init__()
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(dglnn.GraphConv(in_feats, h_feats, activation=self.act))
        for i in range(num_layers-1):
            self.layers.append(dglnn.GraphConv(h_feats, h_feats, activation=self.act))
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        h = self.mlp(h, False)
        return h

class MLP(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, dropout_rate=0, activation='ReLU', **kwargs):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        if num_layers == 0:
            return
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, num_classes))
        else:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for i in range(1, num_layers-1):
                self.layers.append(nn.Linear(h_feats, h_feats))
            self.layers.append(nn.Linear(h_feats, num_classes))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, h, is_graph=True):
        if is_graph:
            h = h.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers)-1:
                h = self.act(h)
        return h


class MGADN(nn.Module):
    def __init__(self, in_feats, h_feats=64, n_head=8, n_layers=4, dropout_rate=0., **kwargs):
        super().__init__()
        self.attn_fn = nn.Tanh()
        self.act_fn = nn.ReLU()
        self.decoder = Decoder(in_feats=in_feats, h_feats=h_feats, n_head=n_head, dropout_rate=0., n_layers=n_layers)
        self.filters3 = GCN(in_feats, h_feats=h_feats, num_classes=h_feats, num_layers=2, mlp_layers=2, dropout_rate=0., activation='ReLU')
        
        self.DMGNN = EdgeConv(in_feats, out_feat=h_feats)
        
        self.linear1 = nn.Linear(h_feats*2, h_feats)
                                    
        self.linear = nn.Sequential(nn.Linear(h_feats, h_feats),
                                     self.attn_fn,
                                     nn.Linear(h_feats, 2))
        
        self.gate_layer = nn.Linear(h_feats, h_feats)
    
    def forward(self, graph):
        x = graph.ndata['feature']
        h_list = []
        x = x.to(torch.float32)
    
        out1 = self.decoder(x, graph)
        out2 = self.filters3(graph)
        
        F = self.DMGNN(graph,x)

        h_list.append(out1)
        h_list.append(out2)
        
        res = torch.cat((h_list[0], h_list[1]), dim=1)
        output = self.linear1(res)
        
        gate = torch.sigmoid(self.gate_layer(output))
        
        out = gate * output + (1 - gate) * F
        result = self.linear(out)
        return result    