import os
import torch
import torch.optim as optim
import numpy as np
import rlkit.torch.pytorch_util as ptu
from torch import nn as nn
from collections import OrderedDict
import copy
from itertools import product
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.metarl_algorithm import OfflineMetaRLAlgorithm_ours


class MBMetaRL(OfflineMetaRLAlgorithm_ours):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            nets,
            obs_shape,
            action_shape,
            goal_radius=1,
            optimizer_class=optim.Adam,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            nets=nets,
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            goal_radius=goal_radius,
            obs_shape=obs_shape,
            action_shape=action_shape,
            **kwargs
        )


        self.soft_target_tau = kwargs['soft_target_tau']
        self.policy_mean_reg_weight = kwargs['policy_mean_reg_weight']
        self.policy_std_reg_weight = kwargs['policy_std_reg_weight']
        self.policy_pre_activation_weight = kwargs['policy_pre_activation_weight']
        self.recurrent = kwargs['recurrent']
        self.kl_lambda = kwargs['kl_lambda']
        self.use_information_bottleneck = kwargs['use_information_bottleneck']
        self.use_value_penalty = kwargs['use_value_penalty']
        self.alpha_max = kwargs['alpha_max']
        self._c_iter = kwargs['c_iter']
        self.train_alpha = kwargs['train_alpha']
        self._target_divergence = kwargs['target_divergence']
        self.alpha_init = kwargs['alpha_init']
        self.alpha_lr = kwargs['alpha_lr']
        self.z_loss_weight = kwargs['z_loss_weight']
        self.max_entropy = kwargs['max_entropy']
        self.allow_backward_z = kwargs['allow_backward_z']


        self.sparse_rewards = kwargs['sparse_rewards']
        self.meta_policy_lr = kwargs['meta_policy_lr']
        self.policy_lr = kwargs['task_policy_lr']
        self.qf_lr = kwargs['qf_lr']
        self.vf_lr = kwargs['vf_lr']
        self.meta_q_lr = kwargs['meta_q_lr']
        self.use_automatic_beta_tuning = kwargs['use_automatic_beta_tuning']
        self.beta_lr = kwargs['beta_lr']
        self.beta_max = kwargs['beta_max']
        self.beta_init = kwargs['beta_init']
        self.meta_beta = kwargs['meta_beta']
        self.pg_step=kwargs['pg_step']
        self.alpha_p_max=kwargs['alpha_p_max']
        self.model_use = kwargs['model_use']
        self.v_loss = kwargs['v_loss']
        self.test_only = kwargs['test_only']
        self.meta_test_only = kwargs['meta_test_only']
        self.test_only_inner_step=kwargs['test_only_inner_step']


        self.loss = {}
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()

        self.horizon = kwargs['horizon']
        self.data_collection_per_epoch = kwargs['data_collection_per_epoch']
        self.batch_size = kwargs['batch_size']
        self.real_data_ratio = kwargs['real_data_ratio']
        self.use_automatic_entropy_tuning = kwargs['use_automatic_entropy_tuning']
        self.train_inner_step = kwargs['train_inner_step']
        self.test_inner_step = kwargs['test_inner_step']
        self.beta = kwargs['beta']
        self.gamma = kwargs['gamma']
        self.type_q_backup = kwargs['type_q_backup']
        self.q_backup_lmbda = kwargs['q_backup_lmbda']
        self.discrete = kwargs['discrete']
        self.reward_scale = kwargs['reward_scale']
        self.discount = kwargs['discount']
        self.critic_criterion = torch.nn.MSELoss()
        self.num_random = kwargs['num_random']
        self.device = ptu.device
        self.min_q_version = kwargs['min_q_version']
        self.temp = kwargs['temp']
        self.min_q_weight = kwargs['min_q_weight']
        self.lagrange_thresh = kwargs['lagrange_thresh']
        self.explore = kwargs['explore']
        self.soft_target_tau = kwargs['soft_target_tau']
        self.obs_shape = obs_shape
        self.num_updates = kwargs['num_updates']
        self.test_num_updates = kwargs['test_num_updates']
        self.policy_update_delay = kwargs['policy_update_delay']
        self.policy_temp = kwargs['policy_temp']
        self.meta_alpha_p = kwargs['meta_alpha_p']
        self.beta_p = kwargs['beta_p']
        self.sync_step = kwargs['sync_step']
        self.update_target_freq = kwargs['update_target_freq']


        self.agent.meta_policy_optimizer = optimizer_class(self.agent.meta_policy.parameters(), lr=self.meta_policy_lr)
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)

        self.q1_pred_epoch=[]
        self.q2_pred_epoch=[]
        self.train_tasks_epoch=[]


        self._num_steps = 0
        self._visit_num_steps_train = 10
        self._alpha_var = torch.tensor(1.)

        for net in nets:
            self.print_networks(net)

    ###### Torch stuff #####
    @property
    def networks(self):
        return [self.agent] + [self.qf1, self.qf2, self.policy]



    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == "gpu":
            device = ptu.device
        for net in self.networks:
            net.to(device)


    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()

        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    ##### Data handling #####

    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None,...]

        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]


    def sample_sac(self, indices, batch_size, buffer, test):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''

        batch_size = batch_size

        batches = [ptu.np_to_pytorch_batch(buffer.random_batch(indices, batch_size=batch_size))]
        unpacked = [self.unpack_batch(batch) for batch in batches]

        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_batch(self, indices, batch_size, buffer,test):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''


        batches = [ptu.np_to_pytorch_batch(buffer.random_batch(indices, batch_size=batch_size))]

        unpacked = [self.unpack_batch(batch) for batch in batches]

        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]

        return unpacked

    ##### Training #####

    def _do_training(self, indices, epoch, test):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size
        num_updates = self.num_updates
        self.train_tasks_epoch=indices


        for i in range(num_updates):

            self.loss['step'] = self._num_steps

            if self.test_only:
                tasks_qv1, tasks_qv2 = self._take_step_momer(indices, self.eval_buffer, i, epoch, test=test, inner_step=self.test_only_inner_step)
            else:
                tasks_qv1, tasks_qv2 = self._take_step_momer(indices, self.train_buffer, i, epoch, test=test, inner_step=self.train_inner_step)

            self._num_steps += 1

        return tasks_qv1,tasks_qv2


    def _do_adaptation(self, idx, buffer):

        for i in range(self.test_num_updates):
            test_qv1, test_qv2 = self._take_step_momer(idx, buffer, i, epoch=1, test=True, inner_step=self.test_inner_step)
        return test_qv1, test_qv2


    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)


    def _take_step_momer(self, indices, buffer, i, epoch, test, inner_step):
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        task_policy_losses=[]
        tasks_qv1=[]
        tasks_qv2=[]
        tasks_weight=torch.zeros(100)
        tasks_weight_q1=torch.zeros(100)
        tasks_weight_q2=torch.zeros(100)
        real_batch_tasks={}
        model_batch_tasks={}
        f_batch_tasks={}

        for task_i in indices:
            if test:

                if self.test_only is False:

                    self.tasks_pool[task_i].task_policy.load_state_dict(self.agent.meta_policy.state_dict())
                    self.tasks_pool[task_i].task_qf1.load_state_dict(self.qf1.state_dict())
                    self.tasks_pool[task_i].task_qf2.load_state_dict(self.qf2.state_dict())
                    self.tasks_pool[task_i].task_qf1_target.load_state_dict(self.qf1.state_dict())
                    self.tasks_pool[task_i].task_qf2_target.load_state_dict(self.qf2.state_dict())
                    self.tasks_pool[task_i].log_alpha=torch.zeros(1, requires_grad=True, device=ptu.device)

                    if self.use_automatic_beta_tuning:
                        if self.meta_beta:
                            self.tasks_pool[task_i].beta=torch.tensor(sum([self.tasks_pool[indice].beta for indice in self.train_tasks_epoch])/len(self.train_tasks_epoch),
                                                          requires_grad=True, device=ptu.device)
                        else:
                            self.tasks_pool[task_i].beta=torch.tensor(self.beta_init, device=ptu.device, requires_grad=True)
                    if self.lagrange_thresh>0:
                        if self.meta_alpha_p:
                            self.tasks_pool[task_i].log_alpha_prime=torch.tensor(sum([self.tasks_pool[indice].log_alpha_prime.to(ptu.device) for indice in self.train_tasks_epoch])/len(self.train_tasks_epoch),
                                                                     requires_grad=True, device=ptu.device)
                        else:
                            self.tasks_pool[task_i].log_alpha_prime=torch.zeros(1, requires_grad=True, device=ptu.device)


            if self.model_use:

                with torch.no_grad():

                    obs, actions, _rewards, _next_obs, terms = self.sample_sac(task_i, self.data_collection_per_epoch, buffer, test)

                    obs = torch.tensor(obs, device = ptu.device)


                    for t in range(self.horizon):

                        policy_output = self.tasks_pool[task_i].task_policy(obs)
                        action = policy_output[8].sample()


                        obs_action = torch.cat([obs, action], dim=-1)

                        next_obs_dists = self.tasks_pool[task_i].task_transition(obs_action)

                        next_obses = next_obs_dists.sample()

                        rewards = next_obses[:, :, -1:]
                        next_obses = next_obses[:, :, :-1]

                        next_obses_mode = next_obs_dists.mean[:, :, :-1]
                        next_obs_mean = torch.mean(next_obses_mode, dim=0)



                        model_indexes = np.random.randint(0, next_obses.shape[0], size=(obs.shape[0]))

                        next_obs = next_obses[model_indexes, :,:].clamp(-10, 10)
                        reward = rewards[model_indexes, :,:].clamp(-10, 10)

                        dones = torch.zeros_like(reward)


                        self.tasks_pool[task_i].model_buffer.add_sample_batch(obs.cpu(), action.cpu(), reward.cpu(), dones.cpu(),
                            next_obs.cpu(), self.data_collection_per_epoch)


                        obs = next_obs

            self.q1_pred_epoch=[]
            self.q2_pred_epoch=[]

            self.tasks_pool[task_i].task_policy_last.load_state_dict(self.tasks_pool[task_i].task_policy.state_dict())# copy.deepcopy(self.agent.meta_policy)
            for _ in range(inner_step):

                if self.model_use==False:
                    self.real_data_ratio=1.0

                real_batch_size = int(self.batch_size * self.real_data_ratio)

                obs_r, actions_r, rewards_r, next_obs_r, terms_r  = self.sample_batch(task_i, real_batch_size, buffer,test) # sample batch from real dataset
                obs_r = obs_r[0,:,:]
                actions_r = actions_r[0, :, :]

                rewards_r = rewards_r[0, :, :]

                next_obs_r = next_obs_r[0, :, :]
                terms_r = terms_r[0, :, :]
                rewards_r=np.array(rewards_r.cpu()).reshape(-1, 1)
                terms_r=np.array(terms_r.cpu()).reshape(-1, 1)
                rewards_r = torch.from_numpy(rewards_r).float().to(ptu.device)
                terms_r = torch.from_numpy(terms_r).float().to(ptu.device)
                real_batch = [obs_r, actions_r, rewards_r, next_obs_r, terms_r]

                if self.model_use:
                    model_batch_size = self.batch_size - real_batch_size
                    model_batch= self.tasks_pool[task_i].model_buffer.random_batch(model_batch_size)
                    obs_m = torch.from_numpy(model_batch["observations"]).float().to(ptu.device)
                    actions_m = torch.from_numpy(model_batch["actions"]).float().to(ptu.device)
                    rewards_m = torch.from_numpy(model_batch["rewards"]).float().to(ptu.device)
                    terms_m = torch.from_numpy(model_batch["terminals"]).float().to(ptu.device)
                    next_obs_m = torch.from_numpy(model_batch["next_observations"]).float().to(ptu.device)
                    sparse_rewards_m  = torch.from_numpy(model_batch["sparse_rewards"]).float().to(ptu.device)
                    model_batch=[obs_m, actions_m, rewards_m, next_obs_m, terms_m]

                    obs_f = torch.cat((obs_r, obs_m), 0)
                    actions_f = torch.cat((actions_r, actions_m), 0)
                    rewards_f = torch.cat((rewards_r, rewards_m), 0)
                    terms_f = torch.cat((terms_r, terms_m), 0)
                    next_obs_f = torch.cat((next_obs_r, next_obs_m), 0)

                    batch_f = [obs_f,actions_f,rewards_f,next_obs_f,terms_f]
                else:
                    batch_f=real_batch
                    model_batch=real_batch

                self._cql_update(real_batch, model_batch, batch_f, task_i, self.agent.meta_policy, _, epoch, test)


            task_qv1 = sum(self.q1_pred_epoch)/inner_step
            task_qv2 = sum(self.q2_pred_epoch)/inner_step

            tasks_weight[task_i] = 1/max(abs(task_qv1),abs(task_qv2))
            tasks_weight_q1[task_i] = 1/abs(task_qv1)
            tasks_weight_q2[task_i] = 1/abs(task_qv2)

            tasks_qv1.append(task_qv1)
            tasks_qv2.append(task_qv2)


        if test == False and i == self.num_updates-1:

            for t in range(0, self.pg_step):
                task_policy_losses=[]
                for task_i in indices:
                    if self.model_use==False:
                        self.real_data_ratio=1.0
                    real_batch_size = int(self.batch_size * self.real_data_ratio)

                    obs_r, actions_r, rewards_r, next_obs_r, terms_r  = self.sample_batch(task_i, real_batch_size, buffer,test) # sample batch from real dataset
                    obs_r = obs_r[0,:,:]
                    actions_r = actions_r[0, :, :]
                    rewards_r = rewards_r[0, :, :]
                    next_obs_r = next_obs_r[0, :, :]
                    terms_r = terms_r[0, :, :]

                    rewards_r=np.array(rewards_r.cpu()).reshape(-1, 1)
                    terms_r=np.array(terms_r.cpu()).reshape(-1, 1)
                    rewards_r = torch.from_numpy(rewards_r).float().to(ptu.device)
                    terms_r = torch.from_numpy(terms_r).float().to(ptu.device)
                    real_batch = [obs_r, actions_r, rewards_r, next_obs_r, terms_r]

                    if self.model_use:
                        model_batch_size = self.batch_size - real_batch_size
                        model_batch= self.tasks_pool[task_i].model_buffer.random_batch(model_batch_size)
                        obs_m = torch.from_numpy(model_batch["observations"]).float().to(ptu.device)
                        actions_m = torch.from_numpy(model_batch["actions"]).float().to(ptu.device)
                        rewards_m = torch.from_numpy(model_batch["rewards"]).float().to(ptu.device)
                        terms_m = torch.from_numpy(model_batch["terminals"]).float().to(ptu.device)
                        next_obs_m = torch.from_numpy(model_batch["next_observations"]).float().to(ptu.device)
                        sparse_rewards_m  = torch.from_numpy(model_batch["sparse_rewards"]).float().to(ptu.device)
                        model_batch=[obs_m, actions_m, rewards_m, next_obs_m, terms_m]

                        obs_f = torch.cat((obs_r, obs_m), 0)
                        actions_f = torch.cat((actions_r, actions_m), 0)
                        rewards_f = torch.cat((rewards_r, rewards_m), 0)
                        terms_f = torch.cat((terms_r, terms_m), 0)
                        next_obs_f = torch.cat((next_obs_r, next_obs_m), 0)

                        batch_f = [obs_f,actions_f,rewards_f,next_obs_f,terms_f]
                    else:
                        batch_f=real_batch
                        model_batch=real_batch

                    if self.v_loss:
                        task_policy_loss = self.calculate_task_loss_v(real_batch, model_batch, batch_f, task_i)
                    else:
                        task_policy_loss = self.calculate_task_loss(real_batch, model_batch, batch_f, task_i)
                    task_policy_losses.append(task_policy_loss)
                meta_loss = sum(task_policy_losses)/len(indices)

                self.agent.meta_policy_optimizer.zero_grad()
                meta_loss.backward()

                for para in self.agent.meta_policy.parameters():

                    para.grad[para.grad!=para.grad]=0
                    para.grad.data.clamp_(-10, 10)

                self.agent.meta_policy_optimizer.step()


            average_q1 = copy.deepcopy(self.tasks_pool[indices[0]].task_qf1).to(ptu.device)
            average_q2 = copy.deepcopy(self.tasks_pool[indices[0]].task_qf2).to(ptu.device)
            for n, m in zip(average_q1.parameters(),average_q2.parameters()):
                n.data.copy_(n.data)
                m.data.copy_(n.data)
            for task_i in indices[1:]:
                for n,m in zip(average_q1.parameters(), self.tasks_pool[task_i].task_qf1.parameters()):
                    n.data.copy_(n.data + m.data)
                for n,m in zip(average_q2.parameters(), self.tasks_pool[task_i].task_qf2.parameters()):
                    n.data.copy_(n.data + m.data)

            self._update_meta_q(self.qf1, average_q1, len(indices),self.meta_q_lr)
            self._update_meta_q(self.qf2, average_q2, len(indices),self.meta_q_lr)

        return tasks_qv1, tasks_qv2





    def forward_cql(self, obs, task_id, meta_update, reparameterize=True, return_log_prob=True):
        log_prob = None
        if meta_update==0:
            policy_output = self.agent.meta_policy(obs, reparameterize=reparameterize, )
        elif meta_update==1:
            policy_output = self.tasks_pool[task_id].task_policy_last(obs, reparameterize=reparameterize, )
        else:
            policy_output = self.tasks_pool[task_id].task_policy(obs, reparameterize=reparameterize, )
        tanh_normal = policy_output[8]
        action=policy_output[0]
        log_prob=policy_output[3]

        return action, log_prob

    def _get_tensor_values_cql(self, obs, actions, network):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        obs_action_temp= torch.cat([obs_temp, actions], dim=-1)
        preds = network(obs_action_temp)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions_cql(self, obs, task_id, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        new_obs_actions, new_obs_log_pi = network(
            obs_temp, task_id, meta_update=2, reparameterize=True, return_log_prob=True,
        )
        if not self.discrete:
            return new_obs_actions.to(ptu.device), new_obs_log_pi.view(obs.shape[0], num_actions, 1)
        else:
            return new_obs_actions.to(ptu.device), new_obs_log_pi




    def calculate_task_loss(self, real_batch, model_batch,batch_data, task_id):
        obs = batch_data[0]
        actions = batch_data[1]

        real_obs = real_batch[0]
        real_actions = real_batch[1]

        rewards = batch_data[2]
        terminals = batch_data[4]
        next_obs = batch_data[3]

        """
        Policy and Alpha Loss
        """
        new_obs_actions, log_pi = self.forward_cql(obs, task_id, meta_update=2)
        if self.use_automatic_entropy_tuning:
            alpha = self.tasks_pool[task_id].log_alpha.exp().to(ptu.device)
        else:
            alpha=1
        obs_action_new = torch.cat([obs, new_obs_actions], dim=-1)
        q_new_actions = torch.min(
            self.tasks_pool[task_id].task_qf1(obs_action_new),
            self.tasks_pool[task_id].task_qf2(obs_action_new),
        )


        div_meta = self._kl_divergence_meta(obs, self.tasks_pool[task_id].task_policy, self.agent.meta_policy, cal_loss=True)

        div_beha = self._kl_divergence_beha(real_obs, real_actions, self.tasks_pool[task_id].task_policy)


        if self.use_automatic_beta_tuning:
            beta_c = torch.min(torch.max(self.tasks_pool[task_id].beta, 0.1 * torch.ones_like((self.tasks_pool[task_id].beta))),
                       self.beta_max * torch.ones_like(self.tasks_pool[task_id].beta)).detach()
        else:
            beta_c = self.beta


        policy_loss = (alpha * log_pi + beta_c * self.gamma * div_meta + beta_c * (
                    1 - self.gamma) * div_beha- q_new_actions).mean()


        return policy_loss

    def _cql_update(self, real_batch, model_batch, batch_data, task_id, meta_policy,i, epoch, test):
        obs = batch_data[0].cuda()
        actions = batch_data[1].cuda()
        rewards = batch_data[2].cuda()

        terminals = batch_data[4].cuda()

        next_obs = batch_data[3].cuda()


        if self.model_use:
            real_obs = real_batch[0].cuda()
            real_actions = real_batch[1].cuda()
            real_next_obs = real_batch[3].cuda()

            model_obs = model_batch[0].cuda()
            model_actions = model_batch[1].cuda()
            model_next_obs = model_batch[3].cuda()
        else:
            real_obs = obs
            real_actions = actions
            real_next_obs =  next_obs
            model_obs = obs
            model_actions = actions
            model_next_obs = next_obs




        """
        Policy and Alpha Loss
        """

        policy_output = self.tasks_pool[task_id].task_policy(obs)
        new_obs_actions, policy_mean, policy_log_std, log_pi = policy_output[:4]
        policy_std = policy_output[5]


        if self.use_automatic_entropy_tuning:

            alpha_loss = -(self.tasks_pool[task_id].log_alpha.cuda() *
                           (log_pi + self.tasks_pool[task_id].target_entropy).detach()).mean()
            self.tasks_pool[task_id].alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.tasks_pool[task_id].alpha_optimizer.step()
            alpha = self.tasks_pool[task_id].log_alpha.exp().cuda()
            alpha = torch.clamp(self.tasks_pool[task_id].log_alpha.exp(), min=0.0, max=1000000.0).to(ptu.device)

        else:
            alpha_loss = 0
            alpha = 1



        obs_action_new = torch.cat([obs, new_obs_actions.cuda()], dim=-1).cuda()

        q_new_actions = torch.min(
            self.tasks_pool[task_id].task_qf1(obs_action_new),
            self.tasks_pool[task_id].task_qf2(obs_action_new),
        )

        if i % self.policy_update_delay == 0:
            div_meta = self._kl_divergence_meta(obs, self.tasks_pool[task_id].task_policy, self.agent.meta_policy, cal_loss=False)

            div_beha = self._kl_divergence_beha(real_obs, real_actions, self.tasks_pool[task_id].task_policy)

            if self.test_only is True:
                if self.meta_test_only is False and self.gamma!=0:
                    self.beta = 0.0
                    self.use_automatic_beta_tuning = False


            if self.use_automatic_beta_tuning:
                if self.gamma ==0:
                    beta_loss = -torch.mean(self.tasks_pool[task_id].beta * (div_beha - self._target_divergence).detach())
                else:

                    beta_loss = -torch.mean(self.tasks_pool[task_id].beta * self.gamma* (div_meta - self._target_divergence).detach()
                                          +self.tasks_pool[task_id].beta *(1- self.gamma)* (div_beha - self._target_divergence).detach())

                beta_loss.backward()
                with torch.no_grad():
                    self.tasks_pool[task_id].beta -= self.beta_lr * self.tasks_pool[task_id].beta.grad
                    self.tasks_pool[task_id].beta.grad.zero_()
                beta_c = torch.min(torch.max(self.tasks_pool[task_id].beta, 0.1 * torch.ones_like((self.tasks_pool[task_id].beta))),
                               self.beta_max * torch.ones_like(self.tasks_pool[task_id].beta))

                policy_loss_1 = (alpha* log_pi + beta_c.detach() * (1-self.gamma) * div_beha.cuda()- q_new_actions.cuda()).mean()
                policy_loss_2 = (beta_c.detach() * self.gamma * div_meta.cuda()).mean()


            else:
                beta_c = self.beta

                policy_loss_1 = (alpha* log_pi + beta_c * (1-self.gamma) * div_beha.cuda()- q_new_actions.cuda()).mean()
                policy_loss_2 = (beta_c * self.gamma * div_meta.cuda()).mean()


            policy_loss = policy_loss_1 + policy_loss_2



            self.tasks_pool[task_id].task_policy_optimizer.zero_grad()
            for para in self.agent.meta_policy.parameters():
                para.requires_grad = False

            policy_loss.backward()
            for para in self.agent.meta_policy.parameters():
                if para.grad is not None:
                    para.grad.zero_()

            self.tasks_pool[task_id].task_policy_optimizer.step()


        """
        QF Loss
        """
        obs_action = torch.cat([obs, actions], dim=-1)
        q1_pred = self.tasks_pool[task_id].task_qf1(obs_action)
        q2_pred = self.tasks_pool[task_id].task_qf2(obs_action)

        new_next_actions, new_log_pi = self.forward_cql(
            next_obs, task_id, meta_update=2, reparameterize=True, return_log_prob=True,
        )
        new_curr_actions, new_curr_log_pi = self.forward_cql(
            obs, task_id,meta_update=2, reparameterize=True, return_log_prob=True,
        )

        next_obs_action = torch.cat([next_obs, new_next_actions], dim=-1)

        target_q_value_all=[]
        if self.type_q_backup == "max":
            target_q_values = torch.max(
                self.tasks_pool[task_id].task_qf1_target(next_obs_action),
                self.tasks_pool[task_id].task_qf2_target(next_obs_action),
            )
            target_q_values = target_q_values - alpha * new_log_pi

        elif self.type_q_backup == "min":
            target_q_value_id = torch.min(
                self.tasks_pool[task_id].task_qf1_target(next_obs_action),
                self.tasks_pool[task_id].task_qf2_target(next_obs_action),
            )
            target_q_values = target_q_value_id - alpha * new_log_pi


        elif self.type_q_backup == "medium":
            target_q1_next = self.tasks_pool[task_id].task_qf1_target(next_obs_action)
            target_q2_next = self.tasks_pool[task_id].task_qf2_target(next_obs_action)
            target_q_values = self.q_backup_lmbda * torch.min(target_q1_next, target_q2_next) \
                              + (1 - self.q_backup_lmbda) * torch.max(target_q1_next, target_q2_next)
            target_q_values = target_q_values - alpha * new_log_pi

        else:
            """when using max q backup"""
            next_actions_temp, _ = self._get_policy_actions_cql(next_obs, task_id, num_actions=10, network=self.forward_cql)
            target_qf1_values = self._get_tensor_values_cql(next_obs, next_actions_temp, network=self.tasks_pool[task_id].task_qf1).max(1)[
                0].view(-1, 1).to(ptu.device)
            target_qf2_values = self._get_tensor_values_cql(next_obs, next_actions_temp, network=self.tasks_pool[task_id].task_qf2).max(1)[
                0].view(-1, 1).to(ptu.device)
            target_q_values = torch.min(target_qf1_values, target_qf2_values)


        if self.use_value_penalty:

            beta_p=500.0
            target_q_values = target_q_values - beta_p * self.gamma * div_meta.cuda() - beta_p * (1-self.gamma) * div_beha.cuda()

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values.detach()


        qf1_loss = self.critic_criterion(q1_pred, q_target)
        qf2_loss = self.critic_criterion(q2_pred, q_target)


        model_obs_action = torch.cat([model_obs, model_actions], dim=-1)
        model_q1_pred = self.tasks_pool[task_id].task_qf1(model_obs_action)
        model_q2_pred = self.tasks_pool[task_id].task_qf2(model_obs_action)

        model_new_next_actions, model_new_log_pi = self.forward_cql(
            model_next_obs, task_id,meta_update=2, reparameterize=True, return_log_prob=True,
        )
        model_new_curr_actions, model_new_curr_log_pi = self.forward_cql(
            model_obs, task_id,meta_update=2, reparameterize=True, return_log_prob=True,
        )


        model_random_actions_tensor = torch.FloatTensor(model_q2_pred.shape[0] * self.num_random,
                                                    model_actions.shape[-1]).uniform_(-1, 1).to(ptu.device)
        model_curr_actions_tensor, model_curr_log_pis = self._get_policy_actions_cql(model_obs, task_id, num_actions=self.num_random,
                                                                                 network=self.forward_cql)
        model_new_curr_actions_tensor, model_new_log_pis = self._get_policy_actions_cql(model_next_obs, task_id, num_actions=self.num_random,
                                                                                    network=self.forward_cql)
        model_q1_rand = self._get_tensor_values_cql(model_obs, model_random_actions_tensor, network=self.tasks_pool[task_id].task_qf1)
        model_q2_rand = self._get_tensor_values_cql(model_obs, model_random_actions_tensor, network=self.tasks_pool[task_id].task_qf2)
        model_q1_curr_actions = self._get_tensor_values_cql(model_obs, model_curr_actions_tensor.detach(), network=self.tasks_pool[task_id].task_qf1)
        model_q2_curr_actions = self._get_tensor_values_cql(model_obs, model_curr_actions_tensor.detach(), network=self.tasks_pool[task_id].task_qf2)
        model_q1_next_actions = self._get_tensor_values_cql(model_obs, model_new_curr_actions_tensor.detach(), network=self.tasks_pool[task_id].task_qf1)
        model_q2_next_actions = self._get_tensor_values_cql(model_obs, model_new_curr_actions_tensor.detach(), network=self.tasks_pool[task_id].task_qf2)

        model_cat_q1 = torch.cat([model_q1_rand, model_q1_pred.unsqueeze(1), model_q1_next_actions, model_q1_curr_actions], 1)
        model_cat_q2 = torch.cat([model_q2_rand, model_q2_pred.unsqueeze(1), model_q2_next_actions, model_q2_curr_actions], 1)



        if self.min_q_version == 3:

            random_density = np.log(0.5 ** model_curr_actions_tensor.shape[-1])
            model_cat_q1 = torch.cat(
                [model_q1_rand - random_density, model_q1_next_actions - model_new_log_pis.detach(),
                model_q1_curr_actions - model_curr_log_pis.detach()], 1
            )
            model_cat_q2 = torch.cat(
                [model_q2_rand - random_density, model_q2_next_actions - model_new_log_pis.detach(),
                model_q2_curr_actions - model_curr_log_pis.detach()], 1
            )

        model_min_qf1_loss = torch.logsumexp(model_cat_q1 / self.temp, dim=1, ).mean() * self.min_q_weight * \
                         self.temp
        model_min_qf2_loss = torch.logsumexp(model_cat_q2 / self.temp, dim=1, ).mean() * self.min_q_weight * \
                         self.temp

        real_obs_action = torch.cat([real_obs, real_actions], dim=-1)
        real_q1_pred = self.tasks_pool[task_id].task_qf1(real_obs_action)
        real_q2_pred = self.tasks_pool[task_id].task_qf2(real_obs_action)


        self.q1_pred_epoch.append(model_q1_pred.mean().cpu().detach().numpy())
        self.q2_pred_epoch.append(model_q2_pred.mean().cpu().detach().numpy())



        """Subtract the log likelihood of data"""
        min_qf1_loss = model_min_qf1_loss - real_q1_pred.mean() * self.min_q_weight
        min_qf2_loss = model_min_qf2_loss - real_q2_pred.mean() * self.min_q_weight

        if self.lagrange_thresh >= 0:


            alpha_prime = torch.clamp(self.tasks_pool[task_id].log_alpha_prime.exp(), min=0.0, max=self.alpha_p_max).to(ptu.device) #1e6
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.lagrange_thresh)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.lagrange_thresh)


            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5

            self.tasks_pool[task_id].alpha_prime_optimizer.zero_grad()

            alpha_prime_loss.backward(retain_graph=True)
            self.tasks_pool[task_id].alpha_prime_optimizer.step()




        qf1_loss = 0.5 * qf1_loss + self.explore * min_qf1_loss
        qf2_loss = 0.5 * qf2_loss + self.explore * min_qf2_loss



        """
        Update critic networks
        """

        self.tasks_pool[task_id].task_qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        for para in self.tasks_pool[task_id].task_qf1.parameters():
            para.grad.data.clamp_(-10, 10)
        self.tasks_pool[task_id].task_qf1_optimizer.step()


        self.tasks_pool[task_id].task_qf2_optimizer.zero_grad()
        qf2_loss.backward()
        for para in self.tasks_pool[task_id].task_qf2.parameters():
            para.grad.data.clamp_(-10, 10)
        self.tasks_pool[task_id].task_qf2_optimizer.step()


        """
        Soft Updates target network
        """
        if i+1 % self.update_target_freq ==0:
            self._sync_weight(self.tasks_pool[task_id].task_qf1_target, self.tasks_pool[task_id].task_qf1, self.soft_target_tau)
            self._sync_weight(self.tasks_pool[task_id].task_qf2_target, self.tasks_pool[task_id].task_qf2, self.soft_target_tau)


    def _update_meta_q(self, meta_net, net, length, meta_lr):

        meta_net.to(ptu.device)
        net.to(ptu.device)
        for o, n in zip(meta_net.parameters(), net.parameters()):

            o.data.copy_(o.data * (1.0 - meta_lr) + n.data.detach() * meta_lr/length)



    def _sync_weight(self, net_target, net, soft_target_tau = 5e-3):

        for o, n in zip(net_target.parameters(), net.parameters()):
            o.data.copy_(o.data.cuda() * (1.0 - soft_target_tau) + n.data * soft_target_tau)


    def _kl_divergence_meta(self, state, policy_1, policy_2, cal_loss):

        policy_output1 = policy_1(state)


        policy_output2 = policy_2(state)

        if cal_loss:
            mean1 = policy_output1[1]
            std1 = policy_output1[5]
            mean2 = policy_output2[1]
            std2 = policy_output2[5]

        else:
            mean1 = policy_output1[1]
            std1 = policy_output1[5]
            mean2 = policy_output2[1]
            std2 = policy_output2[5]

        kl_matrix = ((torch.log(std2 / std1)) + 0.5 * (std1.pow(2)
                                                       + (mean1 - mean2).pow(2)) / std2.pow(2) - 0.5)


        return kl_matrix.sum(1).mean()

    def _kl_divergence_beha(self, state, action, policy_1):
        policy_output = policy_1(state)

        distri = policy_output[8]
        kl_matrix =  distri.log_prob(action)

        return - kl_matrix.sum(dim=1, keepdim=True).mean()


    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            meta_policy=self.agent.meta_policy.state_dict(),
        )
        return snapshot

    def load_epoch_model(self, epoch, log_dir):
        path = log_dir
        try:
            self.agent.meta_policy.load_state_dict(torch.load(os.path.join(path, 'meta_policy_itr_{}.pth'.format(epoch))))
            self.qf1.load_state_dict(torch.load(os.path.join(path, 'qf1_itr_{}.pth'.format(epoch))))
            self.qf2.load_state_dict(torch.load(os.path.join(path, 'qf2_itr_{}.pth'.format(epoch))))
            return True
        except:
            print("epoch: {} is not ready".format(epoch))
            return False
