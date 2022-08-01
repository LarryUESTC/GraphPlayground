import torch
import numpy as np
import pickle as pkl
import scipy.io as sio
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
import os.path as osp
from torch_geometric.datasets import Planetoid, Amazon
import dgl

def load_single_graph(args=None):
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/')
        dataset = Planetoid(path, args.dataset)
    elif args.dataset in ['Photo', 'Computers']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/')
        dataset = Amazon(path, args.dataset, pre_transform=None)  # transform=T.ToSparseTensor(),

    data = dataset[0]
    if args.dataset == 'Cora':

        idx_train = data.train_mask#.ravel()
        idx_val = data.val_mask#.ravel()
        idx_test = data.test_mask#.ravel()
        idx_test[:] = False
        idx_test[range(500, 1500)]=True
    else:
        idx_train = data.train_mask#.ravel()
        idx_val = data.val_mask#.ravel()
        idx_test = data.test_mask#.ravel()


    i = torch.LongTensor([data.edge_index[0].numpy(), data.edge_index[1].numpy()])
    v = torch.FloatTensor(torch.ones([data.num_edges]))
    A_sp = torch.sparse.FloatTensor(i, v, torch.Size([data.num_nodes, data.num_nodes]))
    A = A_sp.to_dense()
    A_nomal = row_normalize(A)
    I = torch.eye(A.shape[1]).to(A.device)
    A_I = A + I
    A_I_nomal = row_normalize(A_I)
    label = data.y


    return [A_I_nomal, A_nomal], data.x, label, idx_train, idx_val, idx_test


def load_acm_mat(sc=3):
    data = sio.loadmat('data/acm.mat')
    label = data['label']

    adj_edge1 = data["PLP"]
    adj_edge2 = data["PAP"]
    adj_fusion1 = adj_edge1 + adj_edge2
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 2] = 0
    adj_fusion[adj_fusion == 2] = 1

    adj1 = data["PLP"] + np.eye(data["PLP"].shape[0])*sc
    adj2 = data["PAP"] + np.eye(data["PAP"].shape[0])*sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_dblp(sc=3):
    data = pkl.load(open("data/dblp.pkl", "rb"))
    label = data['label']

    adj_edge1 = data["PAP"]
    adj_edge2 = data["PPrefP"]
    adj_edge3 = data["PATAP"]
    adj_fusion1 = adj_edge1 + adj_edge2 + adj_edge3
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 3] = 0
    adj_fusion[adj_fusion == 3] = 1

    adj1 = data["PAP"] + np.eye(data["PAP"].shape[0])*sc
    adj2 = data["PPrefP"] + np.eye(data["PPrefP"].shape[0])*sc
    adj3 = data["PATAP"] + np.eye(data["PATAP"].shape[0])*sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3



    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2, adj3]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_imdb(sc=3):
    data = pkl.load(open("data/imdb.pkl", "rb"))
    label = data['label']
###########################################################
    adj_edge1 = data["MDM"]
    adj_edge2 = data["MAM"]
    adj_fusion1 = adj_edge1 + adj_edge2
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 2] = 0
    adj_fusion[adj_fusion == 2] = 1
    ############################################################
    adj1 = data["MDM"] + np.eye(data["MDM"].shape[0])*sc
    adj2 = data["MAM"] + np.eye(data["MAM"].shape[0])*sc
    adj_fusion = adj_fusion  + np.eye(adj_fusion.shape[0])*3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj_fusion = sp.csr_matrix(adj_fusion)


    adj_list = [adj1, adj2]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion

def load_amazon(sc=3):
    data = pkl.load(open("data/amazon.pkl", "rb"))
    label = data['label']

    adj_edge1 = data["IVI"]
    adj_edge2 = data["IBI"]
    adj_edge3 = data["IOI"]
    adj_fusion1 = adj_edge1 + adj_edge2 + adj_edge3
    adj_fusion = adj_fusion1.copy()
    adj_fusion[adj_fusion < 3] = 0
    adj_fusion[adj_fusion == 3] = 1

    adj1 = data["IVI"] + np.eye(data["IVI"].shape[0])*sc
    adj2 = data["IBI"] + np.eye(data["IBI"].shape[0])*sc
    adj3 = data["IOI"] + np.eye(data["IOI"].shape[0])*sc
    adj_fusion = adj_fusion + np.eye(adj_fusion.shape[0]) * 3

    adj1 = sp.csr_matrix(adj1)
    adj2 = sp.csr_matrix(adj2)
    adj3 = sp.csr_matrix(adj3)
    adj_fusion = sp.csr_matrix(adj_fusion)

    adj_list = [adj1, adj2, adj3]

    truefeatures = data['feature'].astype(float)
    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return adj_list, truefeatures, label, idx_train, idx_val, idx_test, adj_fusion


def load_freebase(sc=None):
    type_num = 3492
    ratio = [20, 40, 60]
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "utils/data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_m = sp.eye(type_num)

    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    # pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.FloatTensor(label)
    adj_list = [mam, mdm, mwm]

    # pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]
    return  adj_list, feat_m, label, train[0], val[0], test[0]

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    try:
        return features.todense()
    except:
        return features

def row_normalize(A):
    """Row-normalize sparse matrix"""
    rowsum = A.sum(dim=-1).clamp(min=0.)
    r_inv = rowsum.pow(-1)
    A = r_inv.unsqueeze(-1) * A
    return A

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch2dgl(graph):
    N = graph.shape[0]
    graph_sp = graph.to_sparse()
    edges_src = graph_sp.indices()[0]
    edges_dst = graph_sp.indices()[1]
    edges_features = graph_sp.values()
    graph_dgl = dgl.graph((edges_src,edges_dst), num_nodes = N)
    # graph_dgl.edate['w'] = edges_features
    return graph_dgl

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def mask_edge(graph, mask_prob):
    E = graph.number_of_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx

def RA(graph, x, feat_drop_rate, edge_mask_rate):
    n_node = graph.number_of_nodes()

    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = drop_feature(x, feat_drop_rate)

    ng = dgl.graph([])
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]
    ng.add_edges(nsrc, ndst)

    return ng, feat
