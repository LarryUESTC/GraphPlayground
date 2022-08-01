import torch
from utils import process
from termcolor import cprint

class embedder:
    def __init__(self, args):
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == -1:
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
        cprint("## Loading Dataset ##", "yellow")

        if args.dataset == "dblp":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_dblp(args.sc)
            features = process.preprocess_features(features)
        if args.dataset == "acm":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_acm_mat(args.sc)
            features = process.preprocess_features(features)
        if args.dataset == "imdb":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_imdb(args.sc)
            features = process.preprocess_features(features)
        if args.dataset == "amazon":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_amazon(args.sc)
            features = process.preprocess_features(features)
        if args.dataset == "freebase":
            adj_list, features, labels, idx_train, idx_val, idx_test = process.load_freebase(args.sc)
            features = process.preprocess_features(features)

        adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
        adj_list = [adj.to_dense() for adj in adj_list]
        adj_list = [process.normalize_graph(adj) for adj in adj_list]
        # if args.sparse_adj:
        adj_list = [adj.to_sparse() for adj in adj_list]
        args.nb_nodes = adj_list[0].shape[0]
        args.nb_classes = labels.shape[1]
        args.ft_size = features.shape[1]

        self.adj_list = adj_list
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).to(args.device)
        self.idx_train = torch.LongTensor(idx_train).to(args.device)
        self.idx_val = torch.LongTensor(idx_val).to(args.device)
        self.idx_test = torch.LongTensor(idx_test).to(args.device)

        self.args = args

class embedder_single:
    def __init__(self, args):
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == -1:
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
        cprint("## Loading Dataset ##", "yellow")

        adj_list, features, labels, idx_train, idx_val, idx_test = process.load_single_graph(args)
        features = process.preprocess_features(features)

        # adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
        # adj_list = [adj.to_dense() for adj in adj_list]
        # adj_list = [process.normalize_graph(adj) for adj in adj_list]
        # if args.sparse_adj:
        #     adj_list = [adj.to_sparse() for adj in adj_list]
        args.nb_nodes = adj_list[0].shape[0]
        args.nb_classes = int(labels.max() - labels.min()) + 1
        args.ft_size = features.shape[1]

        self.adj_list = adj_list
        self.dgl_graph = process.torch2dgl(adj_list[0])
        self.features = torch.FloatTensor(features)
        self.labels = labels.to(args.device)
        self.idx_train = idx_train.to(args.device)
        self.idx_val = idx_val.to(args.device)
        self.idx_test = idx_test.to(args.device)

        self.args = args
