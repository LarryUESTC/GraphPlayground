import torch.nn as nn
import numpy as np
import random as random
import torch
import torch.nn as nn
from dgl.nn import GraphConv, EdgeConv, GATConv, SAGEConv, GINConv
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv, ChebConv
# np.random.seed(0)
# torch.backends.cudnn.deterministic = True
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# random.seed(0)


# Applies an average on seq, of shape (nodes, features)
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
            return torch.mean(seq, 0) # 计算各特征在所有结点的均值


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi):
        c_x = c.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)

        logits = torch.cat((sc_1, sc_2))

        return logits


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class GraphConvolution(nn.Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        # support = self.W(input)
        # output= torch.mm(adj, support)
        support = torch.mm(adj, input)
        output = self.W(support)

        return output

def GNN_layer(GNN, d_in, d_out):
    if GNN == 'GCN':
        layer = GraphConv(d_in, d_out)
    elif GNN == 'GAT':
        layer = GATConv(d_in, d_out, num_heads = 5)
    elif GNN == 'SAGE':
        layer = SAGEConv(d_in, d_out, 'mean')
    elif GNN == 'GIN':
        layer = GINConv(torch.nn.Linear(d_in, d_out), 'max', learn_eps = True)
    elif GNN == 'GCN_geometric':
        layer = GCNConv(d_in, d_out)
    elif GNN == 'GCN_org':
        layer = GraphConvolution(d_in, d_out)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % GNN)
    return layer

def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    layers = []
    in_channels = in_channel
    layer_num  = len(cfg)
    for i, v in enumerate(cfg):
        out_channels =  v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
            # result = nn.Sequential(*layers)
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)#, result

class SUGRL_Fast(nn.Module):
    def __init__(self, n_in ,cfg = None, batch_norm=False, act='gelu', dropout = 0.0, final_mlp = 0):
        super(SUGRL_Fast, self).__init__()
        # self.MLP = make_mlplayers(n_in, cfg, batch_norm=False, act=None, dropout = 0, out_layer =None)
        # self.backbone = make_GCNlayers(n_in, cfg, batch_norm=False, act='gelu', dropout = dropout, out_layer =cfg[-1])

        self.dropout = dropout > 0
        self.bat = batch_norm
        self.act = act
        self.final_mlp = final_mlp > 0
        self.A = None
        self.sparse = True
        self.cfg = cfg
        self.layer_num = len(cfg)
        GCN_layers = []
        act_layers = []
        bat_layers = []
        dop_layers = []
        in_channels = n_in
        for i, v in enumerate(cfg):
            out_channels = v
            GCN_layers.append(GraphConv(in_channels, out_channels))
            if act:
                act_layers.append(act_layer(act))
            if batch_norm:
                bat_layers.append(nn.BatchNorm1d(out_channels, affine=False))
            if dropout>0:
                dop_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.GCN_layers = nn.Sequential(*GCN_layers)
        self.act_layers = nn.Sequential(*act_layers)
        self.bat_layers = nn.Sequential(*bat_layers)
        self.dop_layers = nn.Sequential(*dop_layers)
        if self.final_mlp:
            self.mlp = nn.Linear(in_channels, final_mlp)
        else:
            self.mlp = None

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def get_embedding(self, A_a, X_a):
        for i in range(self.layer_num):
            X_a = self.GCN_layers[i](A_a, X_a)
            if self.act:
                X_a = self.act_layers[i](X_a)
            if self.bat:
                X_a = self.bat_layers[i](X_a)
            if self.dropout:
                X_a = self.dop_layers[i](X_a)
        if self.final_mlp:
            embeding_a = self.mlp(X_a)
        else:
            embeding_a = X_a
        return embeding_a.detach()


    def forward(self, A_a, X_a, A_b, X_b):
        X_a = F.dropout(X_a, 0.2)
        X_b = F.dropout(X_b, 0.2)

        for i in range(self.layer_num):
            X_a = self.GCN_layers[i](A_a, X_a)
            X_b = self.GCN_layers[i](A_b, X_b)
            if self.act:
                X_a = self.act_layers[i](X_a)
                X_b = self.act_layers[i](X_b)
            if self.bat:
                X_a = self.bat_layers[i](X_a)
                X_b = self.bat_layers[i](X_b)
            if self.dropout:
                X_a = self.dop_layers[i](X_a)
                X_b = self.dop_layers[i](X_b)
        if self.final_mlp:
            embeding_a = self.mlp(X_a)
            embeding_b = self.mlp(X_b)
        else:
            embeding_a = X_a
            embeding_b = X_b

        embeding_a = (embeding_a - embeding_a.mean(0)) / embeding_a.std(0)
        embeding_b = (embeding_b - embeding_b.mean(0)) / embeding_b.std(0)

        return embeding_a, embeding_b

class GCN_Fast(nn.Module):
    def __init__(self, n_in ,cfg = None, batch_norm=False, act='relu', gnn='GCN', dropout = 0.0, final_mlp = 0):
        super(GCN_Fast, self).__init__()
        # self.MLP = make_mlplayers(n_in, cfg, batch_norm=False, act=None, dropout = 0, out_layer =None)
        # self.backbone = make_GCNlayers(n_in, cfg, batch_norm=False, act='gelu', dropout = dropout, out_layer =cfg[-1])

        self.dropout = dropout
        self.bat = batch_norm
        self.act = act
        self.gnn = gnn
        self.final_mlp = final_mlp > 0
        self.A = None
        self.sparse = True
        self.cfg = cfg
        self.layer_num = len(cfg)
        GCN_layers = []
        act_layers = []
        bat_layers = []
        in_channels = n_in
        for i, v in enumerate(cfg):
            out_channels = v
            GCN_layers.append(GNN_layer(gnn, in_channels, out_channels))
            if act:
                act_layers.append(act_layer(act))
            if batch_norm:
                bat_layers.append(nn.BatchNorm1d(out_channels, affine=False))
            in_channels = out_channels

        self.GCN_layers = nn.Sequential(*GCN_layers)
        self.act_layers = nn.Sequential(*act_layers)
        self.bat_layers = nn.Sequential(*bat_layers)
        # self.dop_layers = nn.Sequential(*dop_layers)
        if self.final_mlp:
            self.mlp = nn.Linear(in_channels, final_mlp)
        else:
            self.mlp = None

        # for m in self.modules():
        #     self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        if isinstance(m, nn.Sequential):
            for mm in m:
                try:
                    torch.nn.init.xavier_uniform_(mm.weight.data)
                except:
                    pass


    def get_embedding(self, A_a, X_a):
        for i in range(self.layer_num):
            X_a = self.GCN_layers[i](A_a, X_a)
            if self.act:
                X_a = self.act_layers[i](X_a)
            if self.bat:
                X_a = self.bat_layers[i](X_a)

        if self.final_mlp:
            embeding_a = self.mlp(X_a)
        else:
            embeding_a = X_a
        return embeding_a.detach()


    def forward(self, A_a, X_a):
        # X_a = F.dropout(X_a, 0.2)
        # X_a = F.dropout(X_a, self.dropout, training=self.training)

        for i in range(self.layer_num):
            X_a = self.GCN_layers[i](A_a, X_a)
            if self.gnn == "GAT":
                X_a = torch.mean(X_a, dim=1)
            if self.act and i != self.layer_num-1:
                X_a = self.act_layers[i](X_a)
            if self.bat and i != self.layer_num-1:
                X_a = self.bat_layers[i](X_a)
            if self.dropout > 0 and i != self.layer_num-1:
                X_a = F.dropout(X_a, self.dropout, training=self.training)
        if self.final_mlp:
            embeding_a = self.mlp(X_a)
        else:
            embeding_a = X_a
        # X_a = F.relu(self.GCN_layers[0](A_a, X_a))
        # embeding_a = self.GCN_layers[1](A_a, X_a)
        # inputx = F.dropout(X_a, 0.1, training=self.training)
        # x = self.GCN_layers[0](A_a, inputx)
        # x = F.relu(x)
        # x = F.dropout(x, 0.1, training=self.training)
        # embeding_a = self.GCN_layers[1](A_a, x)
        # embeding_a = (embeding_a - embeding_a.mean(0)) / embeding_a.std(0)

        return embeding_a

class Action_Net(nn.Module):
    def __init__(self, n_in, cfg=None, batch_norm=False, act='relu', gnn='GCN', dropout=0.0, final_mlp=0):
        super(Action_Net, self).__init__()
        self.dropout = dropout
        self.bat = batch_norm
        self.act = act
        self.gnn = gnn
        self.final_mlp = final_mlp > 0
        self.A = None
        self.sparse = True
        self.cfg = cfg
        self.layer_num = len(cfg)
        GCN_layers = []
        act_layers = []
        bat_layers = []
        in_channels = n_in
        for i, v in enumerate(cfg):
            out_channels = v
            GCN_layers.append(GNN_layer(gnn, in_channels, out_channels))
            if act:
                act_layers.append(act_layer(act))
            if batch_norm:
                bat_layers.append(nn.BatchNorm1d(out_channels, affine=False))
            in_channels = out_channels

        self.GCN_layers = nn.Sequential(*GCN_layers)
        self.act_layers = nn.Sequential(*act_layers)
        self.bat_layers = nn.Sequential(*bat_layers)
        if self.final_mlp:
            self.mlp = nn.Linear(in_channels, final_mlp)
        else:
            self.mlp = None

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        if isinstance(m, nn.Sequential):
            for mm in m:
                try:
                    torch.nn.init.xavier_uniform_(mm.weight.data)
                except:
                    pass

    def forward(self, A_a, X_a):
        # X_a = F.dropout(X_a, self.dropout, training=self.training)
        for i in range(self.layer_num):
            X_a = self.GCN_layers[i](A_a, X_a)
            if self.gnn == "GAT":
                X_a = torch.mean(X_a, dim=1)
            if self.act and i != self.layer_num - 1:
                X_a = self.act_layers[i](X_a)
            if self.bat and i != self.layer_num - 1:
                X_a = self.bat_layers[i](X_a)
            if self.dropout > 0 and i != self.layer_num - 1:
                X_a = F.dropout(X_a, self.dropout, training=self.training)
        if self.final_mlp:
            self.embedding = self.mlp(X_a)
        else:
            self.embedding = X_a

        return  F.log_softmax(self.embedding, dim=1)

class Env_Net(torch.nn.Module):
    def __init__(self, max_layer, nin, cfg):
        self.hidden = []
        super(Env_Net, self).__init__()
        self.conv1 = GraphConvolution(nin, cfg[0])
        for i in range(max_layer - 2):
            self.hidden.append(GraphConvolution(cfg[0], cfg[0]))
        self.conv2 = GraphConvolution(cfg[0], cfg[-1])

        self.hidden = nn.Sequential(*self.hidden)

    def forward(self, action, x, adj):

        x = F.relu(self.conv1(adj, x))
        for i in range(action-2):
            x = F.relu(self.hidden[i](adj, x))
            x = F.dropout(x, training=self.training)
        self.embedding = self.conv2(adj, x)
        return F.log_softmax(self.embedding, dim=1)


class Env_Net_RLG(nn.Module):
    def __init__(self, n_in ,cfg = None, batch_norm=False, act='relu', gnn='GCN_org', dropout = 0.0, final_mlp = 0):
        super(Env_Net_RLG, self).__init__()
        # self.MLP = make_mlplayers(n_in, cfg, batch_norm=False, act=None, dropout = 0, out_layer =None)
        # self.backbone = make_GCNlayers(n_in, cfg, batch_norm=False, act='gelu', dropout = dropout, out_layer =cfg[-1])

        self.dropout = dropout
        self.bat = batch_norm
        self.act = act
        self.gnn = gnn
        self.final_mlp = final_mlp > 0
        self.A = None
        self.sparse = True
        self.cfg = cfg
        self.layer_num = len(cfg)
        GCN_layers = []
        act_layers = []
        bat_layers = []
        in_channels = n_in
        for i, v in enumerate(cfg):
            out_channels = v
            GCN_layers.append(GNN_layer(gnn, in_channels, out_channels))
            if act:
                act_layers.append(act_layer(act))
            if batch_norm:
                bat_layers.append(nn.BatchNorm1d(out_channels, affine=False))
            in_channels = out_channels

        self.GCN_layers = nn.Sequential(*GCN_layers)
        self.act_layers = nn.Sequential(*act_layers)
        self.bat_layers = nn.Sequential(*bat_layers)
        # self.dop_layers = nn.Sequential(*dop_layers)
        if self.final_mlp:
            self.mlp = nn.Linear(in_channels, final_mlp)
        else:
            self.mlp = None

        # for m in self.modules():
        #     self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        if isinstance(m, nn.Sequential):
            for mm in m:
                try:
                    torch.nn.init.xavier_uniform_(mm.weight.data)
                except:
                    pass


    def get_embedding(self, A_a, X_a):
        for i in range(self.layer_num):
            X_a = self.GCN_layers[i](A_a, X_a)
            if self.act:
                X_a = self.act_layers[i](X_a)
            if self.bat:
                X_a = self.bat_layers[i](X_a)

        if self.final_mlp:
            embeding_a = self.mlp(X_a)
        else:
            embeding_a = X_a
        return embeding_a.detach()


    def forward(self, A_a, X_a):
        # X_a = F.dropout(X_a, 0.2)
        # X_a = F.dropout(X_a, self.dropout, training=self.training)

        for i in range(self.layer_num):
            X_a = self.GCN_layers[i](A_a, X_a)
            if self.gnn == "GAT":
                X_a = torch.mean(X_a, dim=1)
            if self.act and i != self.layer_num-1:
                X_a = self.act_layers[i](X_a)
            if self.bat and i != self.layer_num-1:
                X_a = self.bat_layers[i](X_a)
            if self.dropout > 0 and i != self.layer_num-1:
                X_a = F.dropout(X_a, self.dropout, training=self.training)
        if self.final_mlp:
            embeding_a = self.mlp(X_a)
        else:
            embeding_a = X_a
        # X_a = F.relu(self.GCN_layers[0](A_a, X_a))
        # embeding_a = self.GCN_layers[1](A_a, X_a)
        # inputx = F.dropout(X_a, 0.1, training=self.training)
        # x = self.GCN_layers[0](A_a, inputx)
        # x = F.relu(x)
        # x = F.dropout(x, 0.1, training=self.training)
        # embeding_a = self.GCN_layers[1](A_a, x)
        # embeding_a = (embeding_a - embeding_a.mean(0)) / embeding_a.std(0)

        return embeding_a

