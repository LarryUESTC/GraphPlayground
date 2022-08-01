import torch.nn as nn
import random as random
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise, f1_score
from munkres import Munkres
from sklearn import metrics
import numpy as np
import torch
from models.Basic.logreg import LogReg


np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

def evaluate(embeds, idx_train, idx_val, idx_test, labels, seed,epoch,lr ,isTest=True, iterater=10):

    try:
        nb_classes = labels.shape[1]
        train_lbls = torch.argmax(labels[idx_train], dim=1)
        val_lbls = torch.argmax(labels[idx_val], dim=1)
        test_lbls = torch.argmax(labels[idx_test], dim=1)
    except:
        nb_classes = (labels.max() - labels.min() +1).item()
        train_lbls = labels[idx_train]
        val_lbls = labels[idx_val]
        test_lbls = labels[idx_test]

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]
    xent = nn.CrossEntropyLoss()



    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []

    for _ in range(iterater):
        log = LogReg(train_embs.shape[1], nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr)
        log.to(train_lbls.device)

        val_accs = []; test_accs = []
        val_micro_f1s = []; test_micro_f1s = []
        val_macro_f1s = []; test_macro_f1s = []

        for iter_ in range(epoch):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

    if isTest:
        print("\t[Classification] ACC: {:.4f} ({:.4f}) | Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(
            np.mean(accs), np.std(accs), np.mean(macro_f1s), np.std(macro_f1s), np.mean(micro_f1s), np.std(micro_f1s)))
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

    test_embs = np.array(test_embs.cpu())
    test_lbls = np.array(test_lbls.cpu())
    try:
        k1, k2 = run_kmeans(test_embs, test_lbls, nb_classes)
    except:
        k1 = 0
        k2 = 0

    try:
        st = run_similarity_search(test_embs, test_lbls)
    except:
        st = 0

    return np.mean(accs)*100, np.std(accs)*100, np.mean(macro_f1s), np.std(macro_f1s), np.mean(micro_f1s), np.std(micro_f1s), k1, k2, st

def evaluate_single(embeds, idx_train, idx_val, idx_test, labels, seed,epoch,lr ,isTest=True, iterater=10):
    nb_classes = labels.shape[1]
    train_embs = embeds[idx_train]
    xent = nn.CrossEntropyLoss()
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

def run_similarity_search(test_embs, test_lbls):

    numRows = test_embs.shape[0]

    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    st = []
    for N in [5, 10, 20, 50, 100]: #
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows,N)
        st.append(str(np.round(np.mean(np.sum((selected_label == original_label), 1) / N), 4)))

    st = ','.join(st)
    print("\t[Similarity] [5,10,20,50,100] : [{}]".format(st))
    st = st.split(', ')
    return st

class clustering_metrics():
    "from https://github.com/Ruiqi-Hu/ARGA"

    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        return acc, nmi, adjscore

def run_kmeans(x, y, k):
    estimator = KMeans(n_clusters=k)

    NMI_list = []
    NMI_list2 = []
    for i in range(1):
        estimator.fit(x)
        y_pred = estimator.predict(x)
        s = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        NMI_list.append(s)
        cm = clustering_metrics(y, y_pred)
        acc, nmi, adjscore = cm.evaluationClusterModelFromLabel()
        NMI_list2.append(acc)
        NMI_list2.append(nmi)
        NMI_list2.append(adjscore)

    mean = np.mean(NMI_list)
    std = np.std(NMI_list)
    print('\t[Clustering] NMI: {:.4f} | {:.4f}'.format(mean, std))
    print('\t[Clustering] acc, nmi, ari: {:.4f} | {:.4f} | {:.4f} '.format(acc, nmi, adjscore))
    return NMI_list, NMI_list2

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)