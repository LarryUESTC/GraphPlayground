import time
from models.embedder import embedder_single
from evaluate import accuracy
from models.Layers import Env_Net_RLG
import numpy as np
import random as random
import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import defaultdict

np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

class gcn_env(object):
    def __init__(self, args, adj, feature, data_split, label, lr=0.01, weight_decay=5e-4,  batch_size=128, policy=""):
        self.device = args.device
        self.args = args
        self.adj = adj
        self.feature = feature.to(self.device)

        self.model = Env_Net_RLG(self.args.ft_size, self.args.cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        self.idx_train, self.idx_val, self.idx_test = zip(*data_split)
        self.train_indexes = np.where(np.array(self.idx_train) == True)[0]
        self.val_indexes = np.where(np.array(self.idx_val) == True)[0]
        self.test_indexes = np.where(np.array(self.idx_test) == True)[0]
        self.label = label
        self.batch_size = len(self.train_indexes) - 1
        self.i = 0
        self.val_acc = 0.0
        self.policy = policy

        self.buffers = defaultdict(list)

        self.past_acc = []
        self.past_loss = []

    def training(self):

        features = self.feature.to(self.args.device)
        graph_org = self.adj.to(self.args.device)

        print("Started training...")

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        xent = nn.CrossEntropyLoss()
        train_lbls = self.label[self.idx_train]
        val_lbls = self.label[self.idx_val]
        test_lbls = self.label[self.idx_test]

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
            if epoch % 5 == 0 and epoch != 0:
                self.model.eval()
                # A_a, X_a = process.RA(graph_org.cpu(), features, 0, 0)
                # A_a = A_a.add_self_loop().to(self.args.device)
                embeds = self.model(graph_org, features)
                train_acc = accuracy(embeds[self.idx_train], train_lbls)
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
        self.past_acc.append(train_acc.item())
        self.past_loss.append(loss.item())

        return output_acc, training_time, stop_epoch

    def evaluing(self, new_graph):
        self.model.eval()
        ################STA|Eval|###############
        train_lbls = self.label[self.idx_train]
        val_lbls = self.label[self.idx_val]
        test_lbls = self.label[self.idx_test]
        self.model.eval()
        # A_a, X_a = process.RA(graph_org.cpu(), features, 0, 0)
        # A_a = A_a.add_self_loop().to(self.args.device)
        embeds = self.model(new_graph, self.feature)
        train_embs = embeds[self.idx_train]
        val_embs = embeds[self.idx_val]
        test_embs = embeds[self.idx_test]

        loss = F.cross_entropy(train_embs, train_lbls)
        train_acc = accuracy(train_embs, train_lbls)
        val_acc = accuracy(val_embs, val_lbls)
        test_acc = accuracy(test_embs, test_lbls)

        return loss

    def reset(self):
        index = self.train_indexes[self.i]
        state = self.feature[index].to('cpu').numpy()
        self.optimizer.zero_grad()
        return state


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

class Agent(object):
    def __init__(self,
                 scope,
                 replay_memory_size=2000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.95,
                 epsilon_start=1.0,
                 epsilon_end=0.2,
                 epsilon_decay_steps=100,
                 batch_size=128,
                 action_num=2,
                 state_shape=None,
                 norm_step=100,
                 mlp_layers=None,
                 learning_rate=0.0005,
                 device=None):

        self.scope = scope
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.action_num = action_num
        self.norm_step = norm_step

        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # Create estimators
        #with tf.variable_scope(scope):
        self.q_estimator = Estimator(action_num=action_num, learning_rate=learning_rate, state_shape=state_shape, \
            mlp_layers=mlp_layers, device=self.device)
        self.target_estimator = Estimator(action_num=action_num, learning_rate=learning_rate, state_shape=state_shape, \
            mlp_layers=mlp_layers, device=self.device)




    def learn(self, env, total_timesteps):
        done = [False]
        next_state_batch = env.reset2()
        trajectories = []
        for t in range(total_timesteps):
            A = self.predict_batch(next_state_batch)
            best_actions = np.random.choice(np.arange(len(A)), p=A, size=next_state_batch.shape[0])
            state_batch = next_state_batch
            next_state_batch, reward_batch, done_batch, debug = env.step2(best_actions) # debug = (val_acc, test_acc)
            trajectories = zip(state_batch, best_actions, reward_batch, next_state_batch, done_batch)
            for each in trajectories:
                self.feed(each)
        loss = self.train()
        return loss, reward_batch, debug

    def train(self):
        ''' Train the network

        Returns:
            loss (float): The loss of the current batch.
        '''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()

        # Calculate best next actions using Q-network (Double DQN)
        q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        best_actions = np.argmax(q_values_next, axis=1)

        # Evaluate best next actions using Target-network (Double DQN)
        q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
            self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

        # Perform gradient descent update
        state_batch = np.array(state_batch)

        loss = self.q_estimator.update(state_batch, action_batch, target_batch)

        # Update the target estimator
        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator = deepcopy(self.q_estimator)

        self.train_t += 1
        return loss


class RLG(embedder_single):
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
        self.agent = Agent(scope='dqn',
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
            inputs = self.env.generator()
            pred_partition, ll = self.agent.action(inputs)
            real_reward = self.env.reward(inputs, pred_partition)
            real_reward = real_reward.float().squeeze(-1)


            pred_reward = self.env.cri_model(inputs)

            cri_loss = mse_loss(pred_reward, real_reward.detach())
            if i % 5 == 0:  # 更新ciritic net，这里每隔五次更新一次，主要训练两个网络容易不稳定，所以这么设置，但是有无关系不大，主要是两个网络没必要实时迭代更新
                cri_optim.zero_grad()
                cri_loss.backward()
                nn.utils.clip_grad_norm_(cri_model.parameters(), max_norm=1., norm_type=2)
                cri_optim.step()
            if cfg.is_lr_Cri_decay:
                cri_lr_scheduler.step()

            adv = real_reward.detach() - pred_reward.detach()  # 更新action net
            act_loss = (10 * adv * ll).mean()  # *10是因为loss太小   #常规的AC网络， 这里应该是max_pi sum_i{Q*p_i} 这里的Q从reward的期望换成了loss
            act_optim.zero_grad()
            act_loss.backward()
            nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=1., norm_type=2)
            act_optim.step()

            # if val_acc > last_val:  # check whether gain improvement on validation set
            #     best_policy = copy.deepcopy(self.agent)  # save the best policy
            # last_val = val_acc
            # print("Training Meta-policy:", i_episode, "Val_Acc:", val_acc, "Avg_reward:", reward)
        return last_val, last_val, last_val

