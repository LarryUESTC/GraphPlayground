from models.embedder import embedder_single
from models.Layers import Env_Net
import numpy as np
import random as random
import torch.nn.functional as F
import torch
from scipy.sparse import csr_matrix
import copy
from models.Rein.dqn_agent_pytorch import DQNAgent
from gym.spaces import Discrete
from gym import spaces
from collections import defaultdict
from utils.process import sparse_mx_to_torch_sparse_tensor

np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)



class gcn_env(object):
    def __init__(self, args, adj, feature, data_split, label, lr=0.01, weight_decay=5e-4, max_layer=5, batch_size=128, policy=""):
        self.device = args.device
        self.args = args
        self.adj = adj
        self.feature = feature.to(self.device)
        self.init_k_hop(max_layer)
        self.model = Env_Net(max_layer, self.args.ft_size, self.args.cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        self.idx_train, self.idx_val, self.idx_test = zip(*data_split)
        self.train_indexes = np.where(np.array(self.idx_train) == True)[0]
        self.val_indexes = np.where(np.array(self.idx_val) == True)[0]
        self.test_indexes = np.where(np.array(self.idx_test) == True)[0]
        self.label = label
        self.batch_size = len(self.train_indexes) - 1
        self.i = 0
        self.val_acc = 0.0
        self._set_action_space(max_layer)
        obs = self.reset()
        self._set_observation_space(obs)
        self.policy = policy
        self.max_layer = max_layer

        # For Experiment #
        self.random = False
        self.gcn = False  # GCN Baseline
        self.enable_skh = True  # only when GCN is false will be useful
        self.enable_dlayer = True
        self.baseline_experience = 50

        # buffers for updating
        # self.buffers = {i: [] for i in range(max_layer)}
        self.buffers = defaultdict(list)
        self.past_performance = [0]

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def init_k_hop(self, max_hop):
        adj = copy.deepcopy(self.adj)
        adj[range(adj.shape[0]),range(adj.shape[0])] = 0.0
        sp_adj = F.normalize(adj, p = 1).numpy()

        sp_adj = csr_matrix(sp_adj)
        dd = sp_adj
        self.adjs = [sparse_mx_to_torch_sparse_tensor(dd).to_dense()]
        for i in range(max_hop):
            dd *= sp_adj
            self.adjs.append(sparse_mx_to_torch_sparse_tensor(dd).to_dense())

    def reset(self):
        index = self.train_indexes[self.i]
        state = self.feature[index].to('cpu').numpy()
        self.optimizer.zero_grad()
        return state

    def _set_action_space(self, _max):
        self.action_num = _max
        self.action_space = Discrete(_max)

    def _set_observation_space(self, observation):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        self.model.train()
        self.optimizer.zero_grad()
        if self.random == True:
            action = random.randint(1, 5)
        # train one step
        index = self.train_indexes[self.i]
        pred = self.model(action, self.feature, self.adj)[index]
        pred = pred.unsqueeze(0)
        y = self.label[index]
        y = y.unsqueeze(0)
        F.nll_loss(pred, y).backward()
        self.optimizer.step()

        # get reward from validation set
        val_acc = self.eval_batch()

        # get next state
        self.i += 1
        self.i = self.i % len(self.train_indexes)
        next_index = self.train_indexes[self.i]
        # next_state = self.data.x[next_index].to('cpu').numpy()
        next_state = self.feature[next_index].numpy()
        if self.i == 0:
            done = True
        else:
            done = False
        return next_state, val_acc, done, "debug"

    def reset2(self):
        start = self.i
        end = (self.i + self.batch_size) % len(self.train_indexes)
        index = self.train_indexes[start:end]
        state = self.feature[index].to('cpu').numpy()
        self.optimizer.zero_grad()
        return state

    def step2(self, actions):
        self.model.train()
        self.optimizer.zero_grad()
        start = self.i
        end = (self.i + self.batch_size) % len(self.train_indexes)
        index = self.train_indexes[start:end]
        done = False
        for act, idx in zip(actions, index):
            if self.gcn == True or self.enable_dlayer == False:
                act = self.max_layer
            self.buffers[act].append(idx)
            if len(self.buffers[act]) >= self.batch_size:
                self.train(act, self.buffers[act])
                self.buffers[act] = []
                done = True
        if self.gcn == True or self.enable_skh == False:
            ### Random ###
            self.i += min((self.i + self.batch_size) % self.batch_size, self.batch_size)
            start = self.i
            end = (self.i + self.batch_size) % len(self.train_indexes)
            index = self.train_indexes[start:end]
        else:
            index = self.stochastic_k_hop(actions, index)
        next_state = self.feature[index].to('cpu').numpy()
        # next_state = self.data.x[index].numpy()
        val_acc_dict = self.eval_batch()
        val_acc = [val_acc_dict[a] for a in actions]
        test_acc = self.test_batch()
        baseline = np.mean(np.array(self.past_performance[-self.baseline_experience:]))
        self.past_performance.extend(val_acc)
        reward = [100 * (each - baseline) for each in val_acc]  # FIXME: Reward Engineering
        r = np.mean(np.array(reward))
        val_acc = np.mean(val_acc)
        return next_state, reward, [done] * self.batch_size, (val_acc, r)

    def stochastic_k_hop(self, actions, index):
        next_batch = []
        for idx, act in zip(index, actions):
            prob = self.adjs[act][idx].numpy()
            cand = np.array([i for i in range(len(prob))])
            next_cand = np.random.choice(cand, p=prob)
            next_batch.append(next_cand)
        return next_batch

    def train(self, action, indexes):
        self.model.train()
        self.adj = self.adj.to(self.device)
        pred = self.model(action, self.feature, self.adj)[indexes]
        y = self.label[indexes]
        F.nll_loss(pred, y).backward()
        self.optimizer.step()

    def eval_batch(self):
        self.model.eval()
        self.adj = self.adj.to(self.device)
        batch_dict = {}

        val_states = self.feature[self.val_indexes].to('cpu').numpy()
        if self.random == True:
            val_acts = np.random.randint(1, 5, len(self.val_indexes))
        elif self.gcn == True or self.enable_dlayer == False:
            val_acts = np.full(len(self.val_indexes), 3)
        else:
            val_acts = self.policy.eval_step(val_states)
        s_a = zip(self.val_indexes, val_acts)
        for i, a in s_a:
            if a not in batch_dict.keys():
                batch_dict[a] = []
            batch_dict[a].append(i)
        # acc = 0.0
        acc = {a: 0.0 for a in range(self.max_layer)}
        for a in batch_dict.keys():
            idx = batch_dict[a]
            logits = self.model(a, self.feature, self.adj)
            pred = logits[idx].max(1)[1]
            # acc += pred.eq(self.data.y[idx]).sum().item() / len(idx)
            acc[a] = pred.eq(self.label[idx]).sum().item() / len(idx)
        # acc = acc / len(batch_dict.keys())
        return acc

    def test_batch(self):
        self.model.eval()
        self.adj = self.adj.to(self.device)
        batch_dict = {}

        val_states = self.feature[self.test_indexes].to('cpu').numpy()
        if self.random == True:
            val_acts = np.random.randint(1, 5, len(self.test_indexes))
        elif self.gcn == True or self.enable_dlayer == False:
            val_acts = np.full(len(self.test_indexes), 3)
        else:
            val_acts = self.policy.eval_step(val_states)
        s_a = zip(self.test_indexes, val_acts)
        for i, a in s_a:
            if a not in batch_dict.keys():
                batch_dict[a] = []
            batch_dict[a].append(i)
        acc = 0.0
        for a in batch_dict.keys():
            idx = batch_dict[a]
            logits = self.model(a, self.feature, self.adj)
            pred = logits[idx].max(1)[1]
            acc += pred.eq(self.label[idx]).sum().item() / len(idx)
        acc = acc / len(batch_dict.keys())
        return acc

    def check(self):
        self.model.eval()

        tr_states = self.feature[self.train_indexes].to('cpu').numpy()
        tr_acts = self.policy.eval_step(tr_states)


        val_states = self.feature[self.val_indexes].to('cpu').numpy()
        val_acts = self.policy.eval_step(val_states)


        test_states = self.feature[self.test_indexes].to('cpu').numpy()
        test_acts = self.policy.eval_step(test_states)

        return (self.train_indexes, tr_states, tr_acts), (self.val_indexes, val_states, val_acts), (
        self.test_indexes, test_states, test_acts)


class RLGL(embedder_single):
    def __init__(self, args):
        embedder_single.__init__(self, args)
        self.args = args
        self.cfg = args.cfg
        nb_classes = (self.labels.max() - self.labels.min() + 1).item()
        self.cfg.append(nb_classes)
        self.graph_org_torch = self.adj_list[0]
        self.features = self.features
        self.action_num = 5
        self.max_episodes = 325
        self.max_timesteps = 10


        self.data_split = zip(self.idx_train.cpu(), self.idx_val.cpu(), self.idx_test.cpu())
        self.env = gcn_env(self.args, self.graph_org_torch, self.features, self.data_split, self.labels)
        self.agent = DQNAgent(scope='dqn',
                         action_num=self.action_num,
                         replay_memory_size=int(1e4),
                         replay_memory_init_size=500,
                         norm_step=200,
                         state_shape= self.args.ft_size,
                         mlp_layers=[32, 64, 128, 64, 32],
                         device= self.args.device
                         )

        self.env.policy = self.agent
    def training(self):
        print("Training Meta-policy on Validation Set")
        last_val = 0
        for i_episode in range(1, self.max_episodes + 1):
            loss, reward, (val_acc, reward) = self.agent.learn(self.env, self.max_timesteps)  # debug = (val_acc, reward)
            if val_acc > last_val:  # check whether gain improvement on validation set
                best_policy = copy.deepcopy(self.agent)  # save the best policy
            last_val = val_acc
            print("Training Meta-policy:", i_episode, "Val_Acc:", val_acc, "Avg_reward:", reward)
        return last_val, last_val, last_val

