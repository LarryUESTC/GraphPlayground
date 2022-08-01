import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
from models.Layers import GNN_layer, act_layer, AvgReadout, Discriminator
from models.embedder import embedder_single
import os
from evaluate import evaluate


class dgi(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(dgi, self).__init__()
        self.gcn = GNN_layer('GCN_org', n_in, n_h)
        self.activation = act_layer(activation, neg_slope=0.25)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj):
        h_1 = self.activation(self.gcn(adj, seq1))
        c = self.read(h_1)
        c = self.sigm(c)
        h_2 = self.activation(self.gcn(adj, seq2))
        ret = self.disc(c, h_1, h_2)
        return ret

    # Detach the return variables
    def embed(self, seq, adj):
        h_1 = self.activation(self.gcn(adj, seq))
        c = self.read(h_1)

        return h_1.detach(), c.detach()


class DGI(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):
        features = self.features.to(self.args.device)
        adj = self.adj_list[0].to(self.args.device)
        print("Started training...")

        model = dgi(self.args.ft_size, self.args.hid_dim, self.args.activation).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        b_xent = nn.BCEWithLogitsLoss()

        tbar = tqdm(range(self.args.nb_epochs))
        cnt_wait = 0
        best = 1e9
        for _ in tbar:
            model.train()
            optimiser.zero_grad()

            idx = np.random.permutation(self.args.nb_nodes)
            shuf_fts = features[idx, :]

            lbl_1 = torch.ones(self.args.nb_nodes, device=self.args.device)
            lbl_2 = torch.zeros(self.args.nb_nodes, device=self.args.device)
            lbl = torch.cat((lbl_1, lbl_2))

            shuf_fts = shuf_fts.to(self.args.device)


            logits = model(features, shuf_fts, adj)

            loss = b_xent(logits, lbl)

            tbar.set_description('Loss:{:.4f}'.format(loss.item()))

            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(model.state_dict(), self.args.save_root + '/best_dgi.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                print('Early stopping!')
                break

            loss.backward()
            optimiser.step()

        print('Loading model....')
        model.load_state_dict(torch.load(self.args.save_root + '/best_dgi.pkl'))
        print("Evaluating...")
        model.eval()
        embeds, _ = model.embed(features, adj)
        acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, k1, k2, st = evaluate(
            embeds, self.idx_train, self.idx_val, self.idx_test, self.labels,
            seed=self.args.seed, epoch=self.args.test_epo, lr=self.args.test_lr)
        return acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, k1, k2, st
