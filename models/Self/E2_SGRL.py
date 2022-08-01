import time
import os
from models.embedder import embedder
from tqdm import tqdm
from evaluate import evaluate
from models.Layers import make_mlplayers
#from models.SUGRL_Fast import SUGRL_Fast
import numpy as np
import random as random
import torch.nn.functional as F
import torch
import torch.nn as nn

np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

class e2_cgrl(nn.Module):
    def __init__(self, n_in ,cfg = None, dropout = 0.2,sparse = True):
        super(e2_cgrl, self).__init__()
        self.MLP = make_mlplayers(n_in, cfg)
        self.dropout = dropout
        self.A = None
        self.sparse = sparse
        self.cfg = cfg

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq_a, adj_list=None):
        if self.A is None:
            self.A = adj_list
        seq_a = F.dropout(seq_a, self.dropout, training=self.training)

        h_a = self.MLP(seq_a)
        h_p_0 = F.dropout(h_a, self.dropout, training=self.training)

        h_p_list = [] # shape [(num_node,feature_dim),....,(num_node,feature_dim)]
        for adj in adj_list:
            if self.sparse:
                h_p = torch.spmm(adj, h_p_0)
                h_p_list.append(h_p)
            else:
                h_p = torch.mm(adj, h_p_0)
                h_p_list.append(h_p)

        # simple average
        h_p_list_unsqu = [ls.unsqueeze(0) for ls in h_p_list]
        h_p_fusion = torch.mean(torch.cat(h_p_list_unsqu), 0)

        return h_a, h_p_list, h_p_fusion

    def embed(self,  seq_a , adj_list=None ):
        h_a = self.MLP(seq_a)
        h_list = []
        for adj in adj_list:
            if self.sparse:
                h_p = torch.spmm(adj, h_a)
                h_list.append(h_p)
            else:
                h_p = torch.mm(adj, h_a)
                h_list.append(h_p)

        # simple average
        h_p_list_unsqu = [ls.unsqueeze(0) for ls in h_list]
        h_fusion = torch.mean(torch.cat(h_p_list_unsqu), 0)

        return h_a.detach(), h_p.detach(), h_fusion.detach()




class E2_SGRL(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.cfg = args.cfg
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):

        features = self.features.to(self.args.device)
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]

        print("Started training...")

        model = e2_cgrl(self.args.ft_size, cfg=self.cfg, dropout=0.2).to(self.args.device)
        my_margin = self.args.margin1
        my_margin_2 = my_margin + self.args.margin2
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr)

        margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)
        lbl_z = torch.tensor([0.]).to(self.args.device)

        model.train()
        start = time.time()
        for _ in tqdm(range(self.args.nb_epochs)):
            optimiser.zero_grad()
            idx_list = []
            for i in range(self.args.neg_num):
                idx_0 = np.random.permutation(self.args.nb_nodes)
                idx_list.append(idx_0)

            h, h_pos_list, z = model(features, adj_list)

            """compute loss"""
            '''step 1:compute positive sample distance'''
            # compute distance between anchor and positive embeddings in Eq8 :d(h,h^+)^2
            s_pos_ls = []
            for i in range(len(h_pos_list)):
                s_pos_ls.append(F.pairwise_distance(h, h_pos_list[i]))

            # compute distance between  anchor and  common embedding in Eq16 :d(h,z)^2
            u_pos_fusion = F.pairwise_distance(h, z)

            '''step 2:compute negative sample distance'''
            # compute distance between anchor and negative embeddings in Eq8 :d(h,h^-)^2
            s_neg_ls = []
            for negative_id in idx_list:
                s_neg_ls.append(F.pairwise_distance(h, h[negative_id]))

            margin_label = -1 * torch.ones_like(s_pos_ls[0])

            loss_s = 0
            loss_u = 0
            loss_c = 0

            '''step 3:compute loss'''
            # compute L_s Eq8 and 17
            for i in range(len(s_pos_ls)):
                for s_neg in s_neg_ls:
                    loss_s += (margin_loss(s_pos_ls[i], s_neg, margin_label)).mean()
            # compute L_c Eq16 and L_u Eq11
            for s_neg in s_neg_ls:
                loss_c += (margin_loss(u_pos_fusion, s_neg, margin_label)).mean()
                loss_u += torch.max((s_neg - u_pos_fusion.detach() - my_margin_2), lbl_z).sum()
            loss_u = loss_u / self.args.neg_num

            loss = loss_s * self.args.w_s \
                   + loss_u * self.args.w_u \
                   + loss_c * self.args.w_c

            loss.backward()
            optimiser.step()

        training_time = time.time() - start
        print("training time:{}s".format(training_time))
        print("Evaluating...")
        model.eval()
        ha, hp, hf = model.embed(features, adj_list)
        acc, acc_std, macro_f1,macro_f1_std, micro_f1,micro_f1_std,k1, k2, st = evaluate(
            hf, self.idx_train, self.idx_val, self.idx_test, self.labels,
            seed=self.args.seed, epoch=self.args.test_epo,lr=self.args.test_lr)
        return acc, acc_std, macro_f1,macro_f1_std, micro_f1,micro_f1_std,k1, k2, st

