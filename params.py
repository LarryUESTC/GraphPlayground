import argparse

################STA|Semi-supervised Task|###############

class Semi(object):
    def __init__(self, method, dataset):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataset', nargs='?', default=dataset)
        self.parser.add_argument('--method', nargs='?', default=method)
        self.parser.add_argument('--task', type=str, default='semi')
        self.parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--patience', type=int, default=40, help='patience for early stopping')
        self.parser.add_argument('--seed', type=int, default=0, help='the seed to use')
        self.parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
        self.parser.add_argument('--random_aug_feature', type=float, default=0.2, help='RA feature')
        self.parser.add_argument('--random_aug_edge', type=float, default=0.2, help='RA graph')
        self.args, _ = self.parser.parse_known_args()

    def replace(self):
        pass

    def get_parse(self):
        return self.args

class Semi_Gcn(Semi):
    def __init__(self, method, dataset):
        super(Semi_Gcn, self).__init__(method, dataset)
        ################STA|add new params here|###############
        # self.parser.add_argument('--random_aug_edge', type=float, default=0.2, help='RA graph')
        ################END|add new params here|###############
        self.args, _ = self.parser.parse_known_args()

        ################STA|replace params here|###############
        self.replace()
        ################END|replace params here|###############

    def replace(self):
        super(Semi_Gcn, self).replace()
        self.args.__setattr__('method', 'Gcn')
        self.args.__setattr__('lr', 0.05)

class Semi_Gcn_Cora(Semi_Gcn):
    def __init__(self, method, dataset):
        super(Semi_Gcn, self).__init__(method, dataset)
        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Semi_Gcn_Cora, self).replace()
        self.args.__setattr__('dataset', 'Cora')
        self.args.__setattr__('lr', 0.01)

################END|Semi-supervised Task|###############




################STA|unsupervised Task |###############

class Unsup(object):
    def __init__(self, method, dataset):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataset', nargs='?', default=dataset)
        self.parser.add_argument('--method', nargs='?', default=method)
        self.parser.add_argument('--task', type=str, default='unsup')
        self.parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')
        self.parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--patience', type=int, default=40, help='patience for early stopping')
        self.parser.add_argument('--seed', type=int, default=0, help='the seed to use')
        self.parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
        self.parser.add_argument('--random_aug_feature', type=float, default=0.2, help='RA feature')
        self.parser.add_argument('--random_aug_edge', type=float, default=0.2, help='RA graph')

        self.parser.add_argument('--cfg', type=int, default=[512, 128], help='hidden dimension')
        self.parser.add_argument('--nb_epochs', type=int, default=400, help='the number of epochs')
        self.parser.add_argument('--test_epo', type=int, default=100, help='test_epo')
        self.parser.add_argument('--test_lr', type=int, default=0.01, help='test_lr')

        self.args, _ = self.parser.parse_known_args()

    def replace(self):
        pass

    def get_parse(self):
        return self.args

class Unsup_E2sgrl(Unsup):
    def __init__(self,  method, dataset):
        super(Unsup_E2sgrl,self).__init__(method, dataset)
        self.parser.add_argument('--sc', type=int, default=3, help='')
        self.parser.add_argument('--neg_num', type=int, default=2, help='the number of negtives')
        self.parser.add_argument('--margin1', type=float, default=0.8, help='')
        self.parser.add_argument('--margin2', type=float, default=0.4, help='')
        self.parser.add_argument('--w_s', type=float, default=10, help='weight of loss L_s')
        self.parser.add_argument('--w_c', type=float, default=10, help='weight of loss L_c')
        #self.parser.add_argument('--w_ms', type=float, default=1, help='weight of loss L_ms')
        self.parser.add_argument('--w_u', type=float, default=1, help='weight of loss L_u')

        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_E2sgrl, self).replace()
        self.args.__setattr__('method', 'E2sgrl')

class Unsup_E2sgrl_Acm(Unsup_E2sgrl):
    def __init__(self, method, dataset):
        super(Unsup_E2sgrl_Acm,self).__init__(method, dataset)

        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_E2sgrl_Acm, self).replace()
        self.args.__setattr__('dataset', 'acm')
        self.args.__setattr__('cfg', [512, 128])
        self.args.__setattr__('lr', 0.001)
        self.args.__setattr__('nb_epochs', 400)
        self.args.__setattr__('test_epo', 100)
        self.args.__setattr__('test_lr', 0.01)
        self.args.__setattr__('neg_num', 6)
        self.args.__setattr__('margin1', 0.6)
        self.args.__setattr__('margin2', 0.1)
        self.args.__setattr__('w_s', 2)
        self.args.__setattr__('w_c', 2)
        self.args.__setattr__('w_u', 10)

class Unsup_E2sgrl_Dblp(Unsup_E2sgrl):
    def __init__(self, method, dataset):
        super(Unsup_E2sgrl_Dblp,self).__init__(method, dataset)

        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_E2sgrl_Dblp, self).replace()
        self.args.__setattr__('dataset', 'dblp')
        self.args.__setattr__('cfg', [512, 128])
        self.args.__setattr__('lr', 0.0001)
        self.args.__setattr__('nb_epochs', 3500)
        self.args.__setattr__('test_epo', 200)
        self.args.__setattr__('test_lr', 0.01)
        self.args.__setattr__('neg_num', 8)
        self.args.__setattr__('margin1', 0.9)
        self.args.__setattr__('margin2', 0.1)
        self.args.__setattr__('w_s', 0.7)
        self.args.__setattr__('w_c', 9.0)
        self.args.__setattr__('w_u', 0.4)

class Unsup_E2sgrl_Imdb(Unsup_E2sgrl):
    def __init__(self, method, dataset):
        super(Unsup_E2sgrl_Imdb,self).__init__(method, dataset)

        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_E2sgrl_Imdb, self).replace()
        self.args.__setattr__('dataset', 'imdb')
        self.args.__setattr__('cfg', [512, 256])
        self.args.__setattr__('lr', 0.0005)
        self.args.__setattr__('nb_epochs', 600)
        self.args.__setattr__('test_epo', 200)
        self.args.__setattr__('test_lr', 0.01)
        self.args.__setattr__('neg_num', 4)
        self.args.__setattr__('margin1', 1.0)
        self.args.__setattr__('margin2', 1.0)
        self.args.__setattr__('w_s', 9)
        self.args.__setattr__('w_c', 2.5)
        self.args.__setattr__('w_u', 2.5)

class Unsup_E2sgrl_Freebase(Unsup_E2sgrl):
    def __init__(self, method, dataset):
        super(Unsup_E2sgrl_Freebase,self).__init__(method, dataset)

        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_E2sgrl_Freebase, self).replace()
        self.args.__setattr__('dataset', 'freebase')
        self.args.__setattr__('cfg', [256, 256])
        self.args.__setattr__('lr', 0.001)
        self.args.__setattr__('nb_epochs', 1000)
        self.args.__setattr__('test_epo', 50)
        self.args.__setattr__('test_lr', 0.01)
        self.args.__setattr__('neg_num', 14)
        self.args.__setattr__('margin1', 0.55)
        self.args.__setattr__('margin2', 0.1)
        self.args.__setattr__('w_s', 7)
        self.args.__setattr__('w_c', 2)
        self.args.__setattr__('w_u', 0.1)


class Unsup_Sugrl(Unsup):
    def __init__(self,  method, dataset):
        super(Unsup_Sugrl,self).__init__(method, dataset)
        self.parser.add_argument("--NN", default=4, type=int,help='number of negative samples')

        self.parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight_decay in adam')
        self.parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
        self.parser.add_argument('--w_loss1', type=float, default=1, help='')
        self.parser.add_argument('--w_loss2', type=float, default=1, help='')
        self.parser.add_argument('--w_loss3', type=float, default=1, help='')
        self.parser.add_argument('--margin1', type=float, default=0.8, help='')
        self.parser.add_argument('--margin2', type=float, default=0.2, help='')


        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_Sugrl, self).replace()
        self.args.__setattr__('method', 'Sugrl')
        self.args.__setattr__('test_lr', 0.01)

class Unsup_Sugrl_Cora(Unsup_Sugrl):
    def __init__(self,  method, dataset):
        super(Unsup_Sugrl_Cora,self).__init__(method, dataset)
        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_Sugrl_Cora, self).replace()
        self.args.__setattr__('lr', 0.005)
        self.args.__setattr__('dataset', 'Cora')
        self.args.__setattr__('nb_epochs', 500)
        self.args.__setattr__('w_loss1', 10)
        self.args.__setattr__('w_loss2', 10)
        self.args.__setattr__('w_loss3', 1)
        self.args.__setattr__('cfg', [512, 128])
        self.args.__setattr__('margin1', 0.8)
        self.args.__setattr__('margin2', 0.2)
        self.args.__setattr__('NN', 4)
        self.args.__setattr__('weight_decay', 0.00005)
        self.args.__setattr__('dropout', 0.3)
        self.args.__setattr__('test_epo', 100)

class Unsup_Sugrl_CiteSeer(Unsup_Sugrl):
    def __init__(self,  method, dataset):
        super(Unsup_Sugrl_CiteSeer,self).__init__(method, dataset)
        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_Sugrl_CiteSeer, self).replace()
        self.args.__setattr__('lr', 0.005)
        self.args.__setattr__('dataset', 'CiteSeer')
        self.args.__setattr__('nb_epochs', 100)
        self.args.__setattr__('w_loss1', 5)
        self.args.__setattr__('w_loss2', 5)
        self.args.__setattr__('w_loss3', 1)
        self.args.__setattr__('cfg', [128])
        self.args.__setattr__('margin1', 0.8)
        self.args.__setattr__('margin2', 0.4)
        self.args.__setattr__('NN', 5)
        self.args.__setattr__('weight_decay', 0.0001)
        self.args.__setattr__('dropout', 0.1)
        self.args.__setattr__('test_epo', 100)

class Unsup_Sugrl_PubMed(Unsup_Sugrl):
    def __init__(self,  method, dataset):
        super(Unsup_Sugrl_PubMed,self).__init__(method, dataset)
        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_Sugrl_PubMed, self).replace()
        self.args.__setattr__('lr', 0.01)
        self.args.__setattr__('dataset', 'PubMed')
        self.args.__setattr__('nb_epochs', 1000)
        self.args.__setattr__('w_loss1', 20)
        self.args.__setattr__('w_loss2', 20)
        self.args.__setattr__('w_loss3', 1)
        self.args.__setattr__('cfg', [512,128])
        self.args.__setattr__('margin1', 0.5)
        self.args.__setattr__('margin2', 0.5)
        self.args.__setattr__('NN', 3)
        self.args.__setattr__('weight_decay', 0.0001)
        self.args.__setattr__('dropout', 0.4)
        self.args.__setattr__('test_epo', 200)

class Unsup_Sugrl_Photo(Unsup_Sugrl):
    def __init__(self,  method, dataset):
        super(Unsup_Sugrl_Photo,self).__init__(method, dataset)
        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_Sugrl_Photo, self).replace()
        self.args.__setattr__('lr', 0.01)
        self.args.__setattr__('dataset', 'Photo')
        self.args.__setattr__('nb_epochs', 1000)
        self.args.__setattr__('w_loss1', 100)
        self.args.__setattr__('w_loss2', 100)
        self.args.__setattr__('w_loss3', 1)
        self.args.__setattr__('cfg', [512,128])
        self.args.__setattr__('margin1', 0.9)
        self.args.__setattr__('margin2', 0.9)
        self.args.__setattr__('NN', 1)
        self.args.__setattr__('weight_decay', 0.0001)
        self.args.__setattr__('dropout', 0.1)
        self.args.__setattr__('test_epo', 200)

class Unsup_Sugrl_Computers(Unsup_Sugrl):
    def __init__(self,  method, dataset):
        super(Unsup_Sugrl_Computers,self).__init__(method, dataset)
        self.args, _ = self.parser.parse_known_args()
        self.replace()

    def replace(self):
        super(Unsup_Sugrl_Computers, self).replace()
        self.args.__setattr__('lr', 0.01)
        self.args.__setattr__('dataset', 'Computers')
        self.args.__setattr__('nb_epochs', 1000)
        self.args.__setattr__('w_loss1', 100)
        self.args.__setattr__('w_loss2', 100)
        self.args.__setattr__('w_loss3', 1)
        self.args.__setattr__('cfg', [512,128])
        self.args.__setattr__('margin1', 0.9)
        self.args.__setattr__('margin2', 0.9)
        self.args.__setattr__('NN', 5)
        self.args.__setattr__('weight_decay', 0.0001)
        self.args.__setattr__('dropout', 0.1)
        self.args.__setattr__('test_epo', 300)

class Unsup_Dgi(Unsup):
    def __init__(self, method, dataset):
        super(Unsup_Dgi,self).__init__(method, dataset)

        self.parser.add_argument('--wd', type=float, default=0.0, help='weight decay in adam')
        self.parser.add_argument('--hid_dim', type=int, default=512, help='hidden dimension')
        self.parser.add_argument('--activation', type=str, default='prelu', help='activation function after gcn')

        self.args, _ = self.parser.parse_known_args()

        self.args.__setattr__('patience', 20)
        self.args.__setattr__('nb_epochs', 10000)
        self.args.__setattr__('dataset', dataset)
        self.args.__setattr__('lr', 0.001)
        self.args.__setattr__('test_epo', 100)
        self.args.__setattr__('test_lr', 0.01)
        self.replace()

    def replace(self):
        super(Unsup_Dgi, self).replace()
        self.args.__setattr__('method', 'Dgi')



################END|unsupervised Task |###############

################STA|Reinforcement Learning|###############
class Rein(object):
    def __init__(self, method, dataset):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataset', nargs='?', default=dataset)
        self.parser.add_argument('--method', nargs='?', default=method)
        self.parser.add_argument('--task', type=str, default='rein')
        self.parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--patience', type=int, default=40, help='patience for early stopping')
        self.parser.add_argument('--seed', type=int, default=0, help='the seed to use')
        self.parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
        self.parser.add_argument('--random_aug_feature', type=float, default=0.2, help='RA feature')
        self.parser.add_argument('--random_aug_edge', type=float, default=0.2, help='RA graph')
        self.args, _ = self.parser.parse_known_args()

    def replace(self):
        pass

    def get_parse(self):
        return self.args

################END|Reinforcement Learning|###############

params_key = {
'Semi': Semi,
'Semi_Gcn': Semi_Gcn,
'Semi_Gcn_Cora': Semi_Gcn_Cora,
'Unsup': Unsup,
'Unsup_E2sgrl': Unsup_E2sgrl,
'Unsup_E2sgrl_Acm': Unsup_E2sgrl_Acm,
'Unsup_E2sgrl_Dblp': Unsup_E2sgrl_Dblp,
'Unsup_E2sgrl_Imdb': Unsup_E2sgrl_Imdb,
'Unsup_E2sgrl_Freebase': Unsup_E2sgrl_Freebase,
'Unsup_Sugrl':Unsup_Sugrl,
'Unsup_Sugrl_Cora':Unsup_Sugrl_Cora,
'Unsup_Sugrl_CiteSeer':Unsup_Sugrl_CiteSeer,
'Unsup_Sugrl_PubMed':Unsup_Sugrl_PubMed,
'Unsup_Sugrl_Photo':Unsup_Sugrl_Photo,
'Unsup_Dgi':Unsup_Dgi,
'Unsup_Sugrl_Computers':Unsup_Sugrl_Computers,
'Rein': Rein,
}

def parse_args(task, method, dataset):

    name_3 = task + '_' + method + '_' + dataset
    name_2 = task + '_' + method
    name_1 = task

    if name_3 in params_key:
        return params_key[name_3](method, dataset).get_parse()
    elif name_2 in params_key:
        return params_key[name_2](method, dataset).get_parse()
    elif name_1 in params_key:
        return params_key[name_1](method, dataset).get_parse()
    else:
        return None

def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)
