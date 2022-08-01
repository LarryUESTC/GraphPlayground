import time
from utils import process
from models.embedder import embedder_single
import os
from evaluate import evaluate, accuracy
from models.Layers import SUGRL_Fast, GNN_Model
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn


class GCN(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.cfg = args.cfg
        # if not os.path.exists(self.args.save_root):
        #     os.makedirs(self.args.save_root)
        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.cfg.append(nb_classes)
        self.model = GNN_Model(self.args.ft_size, cfg=self.cfg, final_mlp = 0, gnn = self.args.gnn, dropout=self.args.random_aug_feature).to(self.args.device)

    def training(self):

        features = self.features.to(self.args.device)
        graph_org = self.dgl_graph.to(self.args.device)
        graph_org_torch = self.adj_list[0].to(self.args.device)
        print("Started training...")


        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay = self.args.wd)
        xent = nn.CrossEntropyLoss()
        train_lbls = self.labels[self.idx_train]
        val_lbls = self.labels[self.idx_val]
        test_lbls = self.labels[self.idx_test]
        
        cnt_wait = 0
        best = 1e-9
        output_acc = 1e-9
        stop_epoch = 0

        start = time.time()
        totalL = []
        # features = F.normalize(features)
        for epoch in range(self.args.nb_epochs):
            self.model.train()
            optimiser.zero_grad()

            # A_a, X_a = process.RA(graph_org.cpu(), features, self.args.random_aug_feature,self.args.random_aug_edge)
            # A_a = A_a.add_self_loop().to(self.args.device)

            embeds = self.model(graph_org, features)
            embeds_preds = torch.argmax(embeds, dim=1)
            
            train_embs = embeds[self.idx_train]
            val_embs = embeds[self.idx_val]
            test_embs = embeds[self.idx_test]

            
            loss = F.cross_entropy(train_embs, train_lbls)

            loss.backward()
            totalL.append(loss.item())
            optimiser.step()

            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0 :
                self.model.eval()
                # A_a, X_a = process.RA(graph_org.cpu(), features, 0, 0)
                # A_a = A_a.add_self_loop().to(self.args.device)
                embeds = self.model(graph_org, features)
                val_acc = accuracy(embeds[self.idx_val], val_lbls)
                test_acc = accuracy(embeds[self.idx_test], test_lbls)
                # print(test_acc.item())
                # early stop
                stop_epoch = epoch
                if val_acc > best:
                    best = val_acc
                    output_acc = test_acc.item()
                    cnt_wait = 0
                    # torch.save(model.state_dict(), 'saved_model/best_{}.pkl'.format(self.args.dataset))
                else:
                    cnt_wait += 1
                if cnt_wait == self.args.patience:
                    # print("Early stopped!")
                    break
            ################END|Eval|###############


        training_time = time.time() - start
        # print("training time:{}s".format(training_time))
        # print("Evaluating...")
        print("\t[Classification] ACC: {:.4f} | stop_epoch: {:}| training_time: {:.4f} ".format(
            output_acc, stop_epoch, training_time))
        return output_acc, training_time, stop_epoch