import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import degree
from models.Net import make_mlplayers
from embedder import embedder_single
from evaluate import evaluate

class sugrl(nn.Module):
    def __init__(self, n_in, cfg=None, dropout=0.2):
        super(sugrl, self).__init__()
        self.MLP = make_mlplayers(n_in, cfg)
        self.act = nn.ReLU()
        self.dropout = dropout
        self.A = None
        self.sparse = True
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq_a, adj=None):
        if self.A is None:
            self.A = adj
        seq_a = F.dropout(seq_a, self.dropout, training=self.training)

        h_a = self.MLP(seq_a)
        h_p_0 = F.dropout(h_a, 0.2, training=self.training)
        if self.sparse:
            h_p = torch.spmm(adj, h_p_0)
        else:
            h_p = torch.mm(adj, h_p_0)
        return h_a, h_p

    def embed(self, seq_a, adj=None):
        h_a = self.MLP(seq_a)
        if self.sparse:
            h_p = torch.spmm(adj, h_a)
        else:
            h_p = torch.mm(adj, h_a)
        return h_a.detach(), h_p.detach()


class SUGRL(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.cfg = args.cfg
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):
        features = self.features.to(self.args.device)
        adj = self.adj_list[0].to(self.args.device)
        adj_sparse = self.adj_list[0].to_sparse().to(self.args.device)
        print("Started training...")
        model = sugrl(self.args.ft_size, cfg=self.args.cfg,dropout=self.args.dropout).to(self.args.device)
        my_margin = self.args.margin1
        my_margin_2 = my_margin + self.args.margin2
        margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)
        num_neg = self.args.NN

        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        lbl_z = torch.tensor([0.]).to(self.args.device)
        A_degree = degree(adj_sparse._indices()[0], self.args.nb_nodes, dtype=int).tolist()
        edge_index = adj_sparse._indices()[1]
        deg_list_2 = []
        deg_list_2.append(0)
        for i in range(self.args.nb_nodes):
            deg_list_2.append(deg_list_2[-1] + A_degree[i])
        idx_p_list = []
        for j in range(1, 101):
            random_list = [deg_list_2[i] + j % A_degree[i] for i in range(self.args.nb_nodes)]
            idx_p = edge_index[random_list]
            idx_p_list.append(idx_p)

        tbar = tqdm(range(self.args.nb_epochs))
        for current_iter, epoch in enumerate(tbar):
            model.train()
            optimiser.zero_grad()
            idx_list = []
            for i in range(num_neg):
                idx_0 = np.random.permutation(self.args.nb_nodes)
                idx_list.append(idx_0)

            h_a, h_p = model(features, adj)

            h_p_1 = (h_a[idx_p_list[epoch % 100]] + h_a[idx_p_list[(epoch + 2) % 100]] + h_a[
                idx_p_list[(epoch + 4) % 100]] + h_a[idx_p_list[(epoch + 6) % 100]] + h_a[
                         idx_p_list[(epoch + 8) % 100]]) / 5
            s_p = F.pairwise_distance(h_a, h_p)
            s_p_1 = F.pairwise_distance(h_a, h_p_1)
            s_n_list = []
            for h_n in idx_list:
                s_n = F.pairwise_distance(h_a, h_a[h_n])
                s_n_list.append(s_n)
            margin_label = -1 * torch.ones_like(s_p)

            loss_mar = 0
            loss_mar_1 = 0
            mask_margin_N = 0
            for s_n in s_n_list:
                loss_mar += (margin_loss(s_p, s_n, margin_label)).mean()
                loss_mar_1 += (margin_loss(s_p_1, s_n, margin_label)).mean()
                mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).sum()
            mask_margin_N = mask_margin_N / num_neg

            loss = loss_mar * self.args.w_loss1 + loss_mar_1 * self.args.w_loss2 + mask_margin_N * self.args.w_loss3
            loss.backward()
            optimiser.step()
            string_1 = " loss_1: {:.3f}||loss_2: {:.3f}||loss_3: {:.3f}||".format(loss_mar.item(), loss_mar_1.item(),
                                                                                  mask_margin_N.item())
            tbar.set_description(string_1)
        print("Evaluating...")
        model.eval()
        h_a, h_p = model.embed(features, adj)
        embs = h_p
        embs = embs / embs.norm(dim=1)[:, None]
        acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, k1, k2, st = evaluate(
            embs, self.idx_train, self.idx_val, self.idx_test, self.labels,
            seed=self.args.seed, epoch=self.args.test_epo, lr=self.args.test_lr)
        return acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, k1, k2, st

