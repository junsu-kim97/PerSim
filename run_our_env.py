import torch
from simulator.simulator import Simulator
from math import ceil
import numpy as np
from tqdm import tqdm
from envs.config import get_environment_config
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from rl_algorithm.td3 import TD3
from rl_algorithm.dqn import DQN
from utils import load_policy, load_samples
import pandas as pd
from mpc import DynamicsFunc, MPC_M
from typing import Tuple
from torch import Tensor
from utils.get_offline import offline_dataset, segment, format_data_from_merpo_style
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs import ENVS
from configs.default import default_config
import os, json
from utils import deep_update_dict

class Dynamics(DynamicsFunc):
    """

    """

    def __init__(self, sim, env, discerte_action, N, unit, device) -> Tuple[Tensor, Tensor]:
        self.env = env
        if discerte_action:
            action_dim = self.env.action_space.n
        else:
            action_dim = self.env.action_space.shape[0]
        state_dim = self.env.observation_space.shape[0]
        env_name = sim.split('_')[0]
        rank = int(sim.split('_')[5])
        lag = int(sim.split('_')[4])
        delta = sim.split('_')[6][:5] == 'delta'
        self.delta = delta
        self.sim = Simulator(N, action_dim, state_dim, rank, device, lags=lag, state_layers=state_layers[env_name],
                             action_layers=action_layers[env_name], continous_action=not discerte_action, delta=delta)
        self.sim.load(f'simulator/trained/{sim}')
        self.N = N
        self.reward_fun = self.env.torch_reward_fn()
        self.done_fn = self.env.torch_done_fn()
        self.unit = unit
        self.device = device

    def step(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:

        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)

        next_states = self.sim.step(states.clone(), actions, self.unit)[0]
        objective_cost = -1 * self.reward_fun(states, actions, next_states)
        dones = self.done_fn(next_states)
        return next_states, objective_cost, dones


def test_simulator(simulator_name, env_name, device, num_episodes, T=50):
    if env_name in discerte_action_envs:
        discerte_action = True
    else:
        discerte_action = False

    # load env_name
    env_config = get_environment_config(env_name)
    N = env_config['number_of_units']
    env = env_config['env']

    if discerte_action:
        action_dim = env().action_space.n
    else:
        action_dim = env().action_space.shape[0]

    state_dim = env().observation_space.shape[0]

    rank = int(simulator_name.split('_')[5])
    lag = int(simulator_name.split('_')[4])
    delta = simulator_name.split('_')[6][:5] == 'delta'
    sim = Simulator(N, action_dim, state_dim, rank, device, lags=lag, continous_action=not discerte_action,
                    state_layers=state_layers[env_name], action_layers=action_layers[env_name], delta=delta)

    sim.load(f'simulator/trained/{simulator_name}')

    # load policy
    policy_name = f'{env_name}_{policy_class[env_name]}_test'
    policy = load_policy(env(), 'policies', policy_name, policy_class[env_name], device)

    # init parameters
    n_episodes = num_episodes
    test_covaraites = env_config['test_env']
    n_t = len(test_covaraites)
    data = []

    # init trajecotry storage
    # state_all = np.zeros([n_t, 2, n_episodes, state_dim+1, T+1])
    indices = np.arange(5)
    j = -1
    for k, unit in zip(indices, test_covaraites[:]):
        j += 1
        MSE = []
        r2 = []
        parameters = dict(zip(env_config['covariates'], unit))
        state_true = np.zeros([n_episodes, state_dim, T + 1])
        state_sim = torch.zeros([n_episodes, state_dim, T + 1]).to(device)
        for i_episode in tqdm(range(n_episodes)):
            env_ = env(**parameters)
            # actions = np.zeros(T+1)
            total_reward = 0
            observation = env_.reset()
            done = False
            i = 0
            while i <= T:
                # random action
                # action = env_.action_space.sample()
                # test policy action
                action = policy.select_action(np.array(observation))
                old_x0 = observation
                if not done:
                    observation, _, done, _ = env_.step(action)
                    state_true[i_episode, :, i] = observation

                    if i > 0:
                        old_x0 = state_sim[i_episode, :, i - 1].reshape(1, -1)
                    else:
                        old_x0 = torch.tensor(old_x0).float().reshape(1, -1).to(device)
                    action = torch.tensor([action]).reshape(1, -1).to(device)
                    # old_x0 = torch.tensor(old_x0).float().reshape(1,-1).to(device)
                    predicted_state = sim.step(old_x0, action, k)[0]
                    # print(predicted_state.shape)
                    state_sim[i_episode, :, i] = predicted_state
                    i += 1

                else:
                    state_sim[i_episode, :, i:] = np.nan
                    state_true[i_episode, :, i:] = np.nan
                    break

            mse = np.sqrt(
                np.nanmean(np.square(state_sim[i_episode, :, :].cpu().detach().numpy() - state_true[i_episode, :, :])))
            r2_ = r2_score(state_true[i_episode, :, :i].T, state_sim[i_episode, :, :i].cpu().detach().numpy().T,
                           multioutput='variance_weighted')
            r2.append(r2_)
            MSE.append(mse)

        data.append([j, np.mean(MSE), np.std(MSE), np.mean(r2), np.median(r2), ])
    data = pd.DataFrame(data, columns=['agent', 'mean', 'std', 'r2_score_mean', 'r2_score_median'])
    print('prediction error results:')
    print(data)
    data.to_csv(f'simulator/results/{simulator_name}_mse_results.csv')


def test_simulators_ens(simulators, env_name, data_test_pred, device, num_episodes, N, env_config, delta=True, lag=1, T=50):
    ## Discrete or continous action?
    if env_name in discerte_action_envs:
        discrete_action = True
    else:
        discrete_action = False

    if discrete_action:
        action_dim = env_config['env']().action_space.n
    else:
        action_dim = env_config['env']().action_space.shape[0]

    traj_test, traj_test_lengths = segment(
        data_test_pred["states"],
        data_test_pred["actions"],
        data_test_pred["rewards"],
        data_test_pred["dones"],
        data_test_pred["task_id"]
    )
    u_data_test, i_data_test, m_data_test, y_data_test = format_data_from_merpo_style(traj_test, N,
                                                                                      device, delta,
                                                                                      discrete_action, action_dim, lag)

    if env_name in discerte_action_envs:
        discerte_action = True
    else:
        discerte_action = False

    # load env_name
    env_config = get_environment_config(env_name)
    # N = env_config['number_of_units']
    env = env_config['env']

    if discerte_action:
        action_dim = env().action_space.n
    else:
        action_dim = env().action_space.shape[0]

    state_dim = env().observation_space.shape[0]
    # load policy
    # policy_name = f'{env_name}_{policy_class[env_name]}_test'
    # policy = load_policy(env(), 'policies', policy_name, policy_class[env_name], device)
    sims = []
    for simulator_name in simulators:
        rank = int(simulator_name.split('_')[5])
        lag = int(simulator_name.split('_')[4])
        delta = simulator_name.split('_')[6][:5] == 'delta'
        sim = Simulator(N, action_dim, state_dim, rank, device, lags=lag, continous_action=not discerte_action,
                        state_layers=state_layers[env_name], action_layers=action_layers[env_name], delta=delta)
        sim.load(f'simulator/trained/{simulator_name}')
        sims.append(sim)

    # init parameters
    n_episodes = num_episodes
    test_covaraites = env_config['test_env']
    n_t = len(test_covaraites)
    data = []
    # init trajecotry storage
    indices = np.arange(5)
    # MSE_all = np.zeros([len(indices), 2, n_episodes])

    MSE = []
    r2 = []
    loss_fn = torch.nn.MSELoss(reduction='mean')

    params = {'batch_size': 1024, 'shuffle': False}
    data = [u_data_test, i_data_test, m_data_test[:, :state_dim * lag], y_data_test]
    data_ = [[data[0][i], data[1][i], data[2][i], data[3][i]] for i in range(len(data[0]))]
    data_loader = torch.utils.data.DataLoader(data_, **params)
    with torch.no_grad():
        for batch in data_loader:
            u_data, i_data, m_data, y_data = batch
            for sim in sims:
                predicted_state = sim.model((u_data, i_data, m_data))
                MSE.append(loss_fn(predicted_state, y_data).cpu().item())
    MSE_all = sum(MSE) / len(MSE)
    # data = pd.DataFrame(data, columns=['agent', 'mean', 'std', 'r2_score_mean', 'r2_score_median'])
    # print(data)
    # data.to_csv(f'simulator/results/{simulator_name}_mse_results_ens.csv')
    print(MSE_all)
    np.save(f'simulator/results/{simulator_name}_metrics.npy', MSE_all)


def format_data(data, number_of_units, device, delta, discerte_action, action_dim, lags=1):
    '''
    Return data formatted for pytorch.
    '''
    U = []
    I = []
    Time = []
    M = []
    Y = []
    state_dim = data[0]['observations'][0].shape[0]
    for trajectory in data[:]:
        metrics_lags = np.zeros([state_dim * lags])
        for t, (action, metrics, metrics_new) in enumerate(zip(trajectory['actions'],
                                                               trajectory['observations'],
                                                               trajectory['next_observations'])):

            metrics_lags[state_dim:] = metrics_lags[:-state_dim]
            metrics_lags[:state_dim] = metrics
            if t + 1 >= lags:
                unit = np.zeros(number_of_units)

                unit[trajectory['unit_info']['id']] = 1
                U.append(unit)
                if discerte_action:
                    a = np.zeros(action_dim)
                    a[action] = 1
                else:
                    a = action
                I.append(a)
                M.append(list(metrics_lags))
                if delta:
                    Y.append(metrics_new - metrics)
                else:
                    Y.append(metrics_new)
    U = np.array(U)
    I = np.array(I)
    M = np.array(M)
    Y = np.array(Y)
    if len(I.shape) == 1:
        I = I.reshape([-1, 1])
    U = torch.from_numpy(U).float()
    I = torch.from_numpy(I).float()
    M = torch.from_numpy(M).float()
    Y = torch.from_numpy(Y).float()
    return U.to(device), I.to(device), M.to(device), Y.to(device)


def train(dataname, data_train, data_test_adapt, N, env, rank, device, delta=True,
          normalize_state=True, normalize_output=True, lag=1, iterations=300, adapt_iterations=30,
          filename=None, debug=False):
    # config env

    ## Discrete or continous action?
    if env in discerte_action_envs:
        discerte_action = True
    else:
        discerte_action = False

    if discerte_action:
        action_dim = env_config['env']().action_space.n
    else:
        action_dim = env_config['env']().action_space.shape[0]

    # load data
    # data_train = load_samples('datasets/' + dataname + '.pkl')
    # N = env_config['number_of_units']
    # u_data, i_data, m_data, y_data = format_data(data_train[:], N, device, delta, discerte_action, action_dim, lag)
    # loss_fn = torch.nn.MSELoss(reduction='mean')
    # state_dim = y_data.shape[1]

    # load data (ours)
    traj_train, traj_train_lengths = segment(
        data_train["states"],
        data_train["actions"],
        data_train["rewards"],
        data_train["dones"],
        data_train["task_id"]
    )
    u_data, i_data, m_data, y_data = format_data_from_merpo_style(traj_train, N,
                                                                  device, delta, discerte_action, action_dim, lag)

    traj_test_adapt, traj_test_adapt_lenghts = segment(
        data_test_adapt["states"],
        data_test_adapt["actions"],
        data_test_adapt["rewards"],
        data_test_adapt["dones"],
        data_test_adapt['task_id']
    )
    u_data_test, i_data_test, m_data_test, y_data_test = format_data_from_merpo_style(traj_test_adapt, N,
                                                                                      device, delta, discerte_action, action_dim, lag)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    # state_dim = y_data.shape[1]
    state_dim = 18 if env in ['walker-rand-params'] else y_data.shape[1]

    # init parameters
    means, stds, means_state, stds_state = None, None, None, None

    if normalize_state:
        means_state = torch.mean(m_data, 0)
        stds_state = torch.std(m_data, 0)

    if normalize_output:
        means = torch.mean(y_data, 0)
        stds = torch.std(y_data, 0)

    MSE = []
    delta = 'delta' if delta else 'no_delta'

    if filename is None:
        filename = f'{dataname}_{lag}_{rank}_{delta}'

    ## train  simulator
    sim = Simulator(N, action_dim, state_dim, rank, device, lags=lag, continous_action=not discerte_action,
                    means_state=means_state, stds_state=stds_state, means=means, stds=stds, delta=delta,
                    state_layers=state_layers[env], action_layers=action_layers[env], batch_size=batch_size[env])
    sim.train([u_data, i_data, m_data[:, :state_dim * lag], y_data], it=iterations, learning_rate=1e-3)

    # finetune
    sim.train([u_data_test, i_data_test, m_data_test[:, :state_dim * lag], y_data_test], it=iterations, learning_rate=1e-4)
    sim.save(f'simulator/trained/{filename}')


def eval_policy(simulators_, env_name, num_evaluations, N, config, seed, device):
    env_config = get_environment_config(env_name)
    if env_name in discerte_action_envs:
        discerte_action = True
    else:
        discerte_action = False

    if discerte_action:
        action_dimension = env_config['env']().action_space.n
    else:
        action_dimension = env_config['env']().action_space.shape[0]

    state_dimension = env_config['env']().observation_space.shape[0]

    res = np.zeros([len(env_config['test_env']) * num_evaluations, 3])
    # N = env_config['number_of_units']
    j = 0
    for tt, tests in enumerate(env_config['test_env'][:]):
        # if tt != 3 and tt!= 3: continue
        # parameters = dict(zip(env_config['covariates'], tests))
        # env = env_config['env'](**parameters)

        variant = default_config
        if config:
            with open(os.path.join(config)) as f:
                exp_params = json.load(f)
            variant = deep_update_dict(exp_params, variant)
        variant['util_params']['gpu_id'] = 0  # gpu

        if 'normalizer.npz' in os.listdir(variant['algo_params']['data_dir']):
            obs_absmax = np.load(os.path.join(variant['algo_params']['data_dir'], 'normalizer.npz'))['abs_max']
            env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']), obs_absmax=obs_absmax)
        else:
            env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
        env.seed(seed)

        th = mpc_parameters[env_name]['time_horizon']
        num_rollouts = mpc_parameters[env_name]['num_rollouts']
        num_elites = 50
        num_iterations = 5
        max_action = mpc_parameters[env_name]['max_action']
        mountainCar = env_name == 'mountainCar'

        simulators = [Dynamics(simulator, env, discerte_action, N, tt, device) for simulator in simulators_]
        mpc = MPC_M(dynamics_func=simulators, state_dimen=state_dimension, action_dimen=action_dimension,
                    time_horizon=th, num_rollouts=num_rollouts, num_elites=num_elites,
                    num_iterations=num_iterations, disceret_action=discerte_action, mountain_car=mountainCar,
                    max_action=max_action)

        sum_reward = 0
        observation = env.reset()
        k = 0
        i = 0
        while k < num_evaluations:
            state = observation

            actions, terminal_reward = mpc.get_actions(torch.tensor(state, device=device))
            action = actions[0]
            if discerte_action:
                action = action.cpu().numpy()[0]
            else:
                action = action.cpu().numpy()

            observation, reward, done, info = env.step(action)

            sum_reward += reward
            #             if mountainCar and terminal_reward < mpc._rollout_function._time_horizon -5:
            #                mpc._rollout_function._time_horizon = int(terminal_reward.item())+5

            if i % 50 == 0:
                print(f"Reward so far for agent {tt} ,  timestep {i} ,  {k}-th episode: {sum_reward}")
            i += 1
            if done:
                i = 0
                print(f"Reward for agent {tt} in the {k}-th episode: {sum_reward}")
                observation = env.reset()
                k += 1
                res[j, :] = [k, tt, sum_reward]
                j += 1
                sum_reward = 0
                mpc = MPC_M(dynamics_func=simulators, state_dimen=state_dimension, action_dimen=action_dimension,
                            time_horizon=th, num_rollouts=num_rollouts, num_elites=num_elites,
                            num_iterations=num_iterations, disceret_action=discerte_action, max_action=max_action,
                            mountain_car=mountainCar)

        env.close()
    res = pd.DataFrame(res, columns=['trial', 'agent', 'reward'])
    res.to_csv(f'simulator/mpc_results/mpc_sim_{simulators_[0]}.csv')


##### Fixed Parameters ####

# TODO: check hyperparameters for walker-rand-params
action_layers = {'mountainCar': [50], 'cartPole': [50], 'halfCheetah': [512, 512], 'walker-rand-params': [512, 512]}
batch_size = {'mountainCar': 512, 'cartPole': 64, 'halfCheetah': 1024, 'walker-rand-params': 1024}
normalize = {'mountainCar': False, 'cartPole': False, 'halfCheetah': True, 'walker-rand-params': True}
state_layers = {'mountainCar': [256], 'cartPole': [256], 'halfCheetah': [512, 512, 512, 512], 'walker-rand-params': [512, 512, 512, 512]}
policy_class = {'mountainCar': 'DQN', 'cartPole': 'DQN', 'halfCheetah': 'TD3', 'slimHumanoid': 'TD3', 'walker-rand-params': 'TD3'}
mpc_parameters = {'mountainCar':
    {
        'time_horizon': 50, 'num_rollouts': 1000, "max_action": None
    },
    'cartPole':
        {
            'time_horizon': 50, 'num_rollouts': 1000, "max_action": None
        },
    'halfCheetah':
        {
            'time_horizon': 30, 'num_rollouts': 200, "max_action": 1
        },
    'walker-rand-params':
        {
            'time_horizon': 30, 'num_rollouts': 200, "max_action": 1
        }
}

discerte_action_envs = {'mountainCar', 'cartPole'}
parser = argparse.ArgumentParser(description='interface of running experiments for  baselines')
parser.add_argument('--env', type=str, default='mountainCar', help='choose envs to generate data for')
parser.add_argument('--dataname', type=str, default='mountainCar_random_0.0_0', help='choose envs to generate data for')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--trial', type=int, default=0, help='trial number')
parser.add_argument('--r', type=int, default=3, help='tensor rank')
parser.add_argument('--lag', type=int, default=1, help='lag')
parser.add_argument('--delta', dest='delta', action='store_true')
parser.add_argument('--no-delta', dest='delta', action='store_false')
parser.add_argument('--normalize_state', dest='normalize_state', action='store_true')
parser.add_argument('--normalize_output', dest='normalize_output', action='store_true')
parser.add_argument('--no_normalize_state', dest='normalize_state', action='store_false')
parser.add_argument('--no_normalize_output', dest='normalize_output', action='store_false')
parser.add_argument('--num_episodes', type=int, default=200, help='gpu device id')
parser.add_argument('--num_mpc_evals', type=int, default=20, help='number of MPC episodes')
parser.add_argument('--num_simulators', type=int, default=5, help='number of models')
parser.add_argument('--iterations', type=int, default=300)
parser.add_argument('--adapt_iterations', type=int, default=300)
parser.add_argument('--config', type=str, default='exp_config/walker_rand_params.json')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--debug', action="store_true")

parser.set_defaults(delta=True)
parser.set_defaults(normalize_state=True)
parser.set_defaults(normalize_output=True)

args = parser.parse_args()

args.normalize_state = normalize[args.env]
args.normalize_output = normalize[args.env]

# set device
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
delta = 'delta' if args.delta else 'no_delta'

torch.manual_seed(args.seed)
np.random.seed(args.seed)

simulators = []

env_config = get_environment_config(args.env)
# TODO: make arguments for data_dir, n_trj, tasks, ratio
data_dir = '/home/junsu/workspace/faster-trajectory-transformer/data/new-walker-rand-params'
N_train = env_config['number_of_train_units']
N_test = env_config['number_of_test_units']
if args.debug:
    N_train = 2
    N_test = 2

N = N_train + N_test

train_tasks = range(0, N_train)
test_tasks = range(N_train, N_train + N_test)
N_train_idxs = env_config['number_of_train_idxs']
N_test_adapt_idxs = env_config['number_of_test_adapt_idxs']
N_test_pred_idxs = env_config['number_of_test_pred_idxs']

train_idxs = range(0, N_train_idxs)
test_adapt_idxs = range(0, N_test_adapt_idxs)
test_pred_idxs = range(N_test_adapt_idxs, N_test_adapt_idxs + N_test_pred_idxs)
data_train = offline_dataset(data_dir=data_dir, tasks=train_tasks, idxs=train_idxs, ratio=1.0)
data_test_adapt = offline_dataset(data_dir=data_dir, tasks=test_tasks, idxs=test_adapt_idxs, ratio=1.0)
data_test_pred = offline_dataset(data_dir=data_dir, tasks=test_tasks, idxs=test_pred_idxs, ratio=1.0)

# train and test simulators
for i in range(args.num_simulators):
    print('==' * 10)
    print(f'Train the {i}-th simulator')
    print('==' * 10)
    filename = f'{args.dataname}_{args.lag}_{args.r}_{delta}_{i}_{args.trial}'
    train(args.dataname, data_train, data_test_adapt, N, args.env, args.r, device, normalize_state=args.normalize_state,
          normalize_output=args.normalize_output, iterations=args.iterations, adapt_iterations=args.adapt_iterations,
          filename=filename, debug=args.debug)
    simulators.append(filename)

# test simulators prediction accuracy
print('==' * 10)
print(f'Test the prediction error for simulators')
print('==' * 10)
test_simulators_ens(simulators, args.env, data_test_pred, device, args.num_episodes, N, env_config, T=50)

print('==' * 10)
print(f'Evaluate Average Reward via MPC')
print('==' * 10)
eval_policy(simulators, args.env, args.num_mpc_evals, N, args.config, args.seed, device)
# Estimate average reward


