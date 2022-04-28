import numpy as np

import torch
import copy
from torch import nn as nn
import torch.optim as optim
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.torch.sac.policies import TanhGaussianPolicy_ours
from rlkit.torch.networks import FlattenMlp, Mlp



def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


# data buffer
class COMBOBuffer:
    def __init__(self, buffer_size):
        self.data = None
        self.buffer_size = int(buffer_size)

    def put(self, batch_data):
        batch_data.to_torch(device='cuda:0') #cpu

        if self.data is None:
            self.data = batch_data
        else:
            self.data.cat_(batch_data)


        if len(self) > self.buffer_size:
            self.data = self.data[len(self) - self.buffer_size:]

    def __len__(self):
        if self.data is None: return 0
        return self.data.shape[0]

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self), size=(batch_size))
        return self.data[indexes]


def soft_clamp(x: torch.Tensor, _min=None, _max=None):

    if _max is not None:
        x = _max.to(ptu.device) - F.softplus(_max.to(ptu.device) - x)
    if _min is not None:
        x = _min.to(ptu.device) + F.softplus(x - _min.to(ptu.device))
    return x


# transition model

class EnsembleLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=7):
        super().__init__()

        self.ensemble_size = ensemble_size

        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(ensemble_size, in_features, out_features,device=ptu.device)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features, device=ptu.device)))

        torch.nn.init.trunc_normal_(self.weight, std=1 / (2 * in_features ** 0.5))

        self.select = list(range(0, self.ensemble_size))

    def forward(self, x):
        weight = self.weight[self.select].to(ptu.device)
        bias = self.bias[self.select].to(ptu.device)



        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('dij,bjk->bik', x, weight)

        x = x + bias

        return x

    def set_select(self, indexes):
        assert len(indexes) <= self.ensemble_size and max(indexes) < self.ensemble_size
        self.select = indexes

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class EnsembleTransition(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_features, hidden_layers, ensemble_size=7, mode='local',
                 with_reward=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.mode = mode
        self.with_reward = with_reward
        self.ensemble_size = ensemble_size

        self.activation = Swish()



        module_list = []
        for i in range(hidden_layers):
            if i == 0:
                module_list.append(EnsembleLinear(obs_dim + action_dim, hidden_features, ensemble_size))
            else:
                module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size))
        self.backbones = torch.nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(hidden_features, 2 * (obs_dim + self.with_reward), ensemble_size)

        self.register_parameter('max_logstd',
                                torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * 1, requires_grad=False))
        self.register_parameter('min_logstd',
                                torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * -5, requires_grad=False))

    def forward(self, obs_action):
        output = obs_action

        for layer in self.backbones:
            output = self.activation(layer(output))

        mu, logstd = torch.chunk(self.output_layer(output), 2, dim=-1)
        logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
        if self.mode == 'local':
            if self.with_reward:
                obs, reward = torch.split(mu, [self.obs_dim, 1], dim=-1)
                obs = obs + obs_action[..., :self.obs_dim]
                mu = torch.cat([obs, reward], dim=-1)
            else:
                mu = mu + obs_action[..., :self.obs_dim]


        return torch.distributions.Normal(mu, torch.exp(logstd))

    def set_select(self, indexes):
        for layer in self.backbones:
            layer.set_select(indexes)
        self.output_layer.set_select(indexes)




class TASKAgent(nn.Module):

    def __init__(self,
                 index,
                 nets,
                 task_para_list,
                 inner_dnns,
                 train_test,
                 optimizer_class=optim.Adam,
                 **kwargs
                 ):
        super().__init__()
        self.index = index


        self.obs_shape = task_para_list[0]
        self.action_shape = task_para_list[1]
        self.hidden_layer_size = task_para_list[2]
        self.transition_layers = task_para_list[3]
        self.transition_init_num = task_para_list[4]

        self.device= ptu.device
        self.transition_lr = task_para_list[6]
        self.use_automatic_entropy_tuning = task_para_list[7]
        self.target_entropy = task_para_list[8]
        self.task_policy_lr = task_para_list[9]
        self.lagrange_thresh = task_para_list[10]
        self.qf_lr = task_para_list[11]
        self.buffer_size = task_para_list[12]
        self.goal_radius = task_para_list[13]
        self.net_size = task_para_list[14]
        self.alpha_lr = task_para_list[17]
        self.alpha_p_lr = task_para_list[17]
        self.use_automatic_beta_tuning = task_para_list[15]
        self.beta_init = task_para_list[16]


        if train_test:
            self.task_transition = inner_dnns[self.index]

        else:
            self.task_transition = inner_dnns[self.index]
            self.qf_lr = task_para_list[18]



        self.task_qf1 = Mlp(
                hidden_sizes=[self.net_size, self.net_size, self.net_size],
                input_size=self.obs_shape +  self.action_shape,
                output_size=1,
            )
        self.task_qf2 = Mlp(
            hidden_sizes=[self.net_size, self.net_size, self.net_size],
            input_size=self.obs_shape + self.action_shape,
            output_size=1,
            )

        self.task_policy = TanhGaussianPolicy_ours(
            hidden_sizes=[self.net_size, self.net_size, self.net_size],
            obs_dim=self.obs_shape,
            action_dim= self.action_shape,
            )

        self.task_policy_last = TanhGaussianPolicy_ours(
            hidden_sizes=[self.net_size, self.net_size, self.net_size],
            obs_dim=self.obs_shape,
            action_dim= self.action_shape,
        )


        self.task_qf1_target = Mlp(
            hidden_sizes=[self.net_size, self.net_size, self.net_size],
            input_size=self.obs_shape +  self.action_shape,
            output_size=1,
        )
        self.task_qf2_target = Mlp(
            hidden_sizes=[self.net_size, self.net_size, self.net_size],
            input_size=self.obs_shape + self.action_shape,
            output_size=1,
        )


        if task_para_list[7]:
            if task_para_list[8]:
                self.target_entropy = task_para_list[8]
            else:
                self.target_entropy = -np.prod(task_para_list[1]).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=self.alpha_lr,
            )

        if task_para_list[15]:
            self.beta = torch.tensor(self.beta_init, device=ptu.device, requires_grad=True)

        if task_para_list[10] >= 0:
            self.target_action_gap = task_para_list[10]
            self.log_alpha_prime = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_prime_optimizer = torch.optim.Adam(
                [self.log_alpha_prime],
                lr=self.alpha_p_lr,
            )

        self.task_qf1_optimizer = torch.optim.Adam(self.task_qf1.parameters(), lr=task_para_list[11])
        self.task_qf2_optimizer = torch.optim.Adam(self.task_qf2.parameters(), lr=task_para_list[11])
        self.task_policy_optimizer = torch.optim.Adam(self.task_policy.parameters(), lr=task_para_list[9])



        self.buffer_size = task_para_list[12]


        self.model_buffer = SimpleReplayBuffer(
            max_replay_buffer_size=self.buffer_size,
            observation_dim=task_para_list[0],
            action_dim=task_para_list[1],
            goal_radius=self.goal_radius,
        )

        for net in self.networks:
            net.to(ptu.device)





    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs], dim=1)
        return self.task_policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, task_indices=None):
        ''' given context, get statistics under the current policy of a set of observations '''


        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)


        in_ = torch.cat([obs], dim=1)
        policy_outputs = self.task_policy(t, b, in_, reparameterize=True, return_log_prob=True)


        return policy_outputs

    def log_diagnostics(self, eval_statistics):

        eval_statistics['task_idx'] = self.task_indices[0]

    @property
    def networks(self):
        return [self.task_qf1, self.task_qf2, self.task_qf1_target,self.task_qf2_target, self.task_policy]

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
