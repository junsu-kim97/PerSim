import os
import glob
import numpy as np
from tqdm import tqdm
from tqdm.auto import trange
import torch


def load_paths(data_dir, tasks, idxs):
    trj_paths = []
    for n in tasks:
        for i in idxs:
            # trj_paths += glob.glob(os.path.join(data_dir, "goal_idx%d" % (n), "trj_evalsample*_step*.npy"))
            trj_paths += glob.glob(os.path.join(data_dir, "goal_idx%d" % (n),  "trj_evalsample%d" % (i) + "_step*.npy"))
    paths = [
        trj_path
        for trj_path in trj_paths
        if int(trj_path.split("/")[-2].split("goal_idx")[-1]) in tasks
    ]
    task_idxs = [
        int(trj_path.split("/")[-2].split("goal_idx")[-1])
        for trj_path in trj_paths
        if int(trj_path.split("/")[-2].split("goal_idx")[-1]) in tasks
    ]
    return paths, task_idxs


def offline_dataset(data_dir, tasks, idxs, ratio=1.0, append_reward_in_obs=True):
    paths, task_idxs = load_paths(data_dir, tasks, idxs)
    if ratio < 1.0:
        indexes = np.random.choice(range(len(paths)), int(len(paths) * ratio), replace=False).tolist()
        paths = [paths[index] for index in indexes]
    obs_, action_, reward_ = [], [], []
    done_, real_done_ = [], []
    task_id_ = []

    for path in tqdm(paths):
        path_tokens = path.split("/")
        for path_token in path_tokens:
            if "goal_idx" in path_token:
                task_id = int(path_token.split("goal_idx")[-1])

        dataset = np.load(path, allow_pickle=True)
        for idx, (obs, action, reward, next_obs) in enumerate(dataset):
            real_done = 1 if idx == len(dataset) - 1 else 0
            done = real_done

            if append_reward_in_obs:
                if idx == 0:
                    # TODO: check for walker intial reward
                    if 'walker' in data_dir or 'hopper' in data_dir:
                        obs_reward = 1.
                    elif 'half-cheetah' in data_dir:
                        obs_reward = 0.
                    else:
                        raise NotImplementedError
                else:
                    obs_reward = dataset[idx - 1][2]
                obs = np.concatenate([np.array(obs_reward).reshape(1), obs])
            obs_.append(obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done)
            real_done_.append(real_done)
            task_id_.append(task_id)

    return {
        "states": np.array(obs_),
        "actions": np.array(action_),
        "rewards": np.array(reward_)[:, None],
        "dones": np.array(done_)[:, None],
        "real_dones": np.array(real_done_)[:, None],
        "task_id": np.array(task_id_)[:, None],
    }

def format_data_from_merpo_style(data, number_of_units, device, delta, discerte_action, action_dim, lags=1):
    '''
    Return data formatted for pytorch.
    '''
    U, I, Time, M, Y = [], [], [], [], []
    state_dim = data[0]['states'][0].shape[0]
    for _, trajectory in tqdm(zip(data.keys(), data.items())):
        trajectory_id, trajectory = trajectory
        metrics_lags = np.zeros([state_dim * lags])
        for t, (action, metrics) in enumerate(zip(trajectory['actions'], trajectory['states'])):
            if t+1 < len(trajectory['states']):
                metrics_new = trajectory['states'][t+1]
            else:
                # metrics_new = trajectory['states'][t]
                break

            metrics_lags[state_dim:] = metrics_lags[:-state_dim]
            metrics_lags[:state_dim] = metrics
            if t + 1 >= lags:
                unit = np.zeros(number_of_units)

                unit[trajectory['task_ids']]= 1
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

def segment(states, actions, rewards, terminals, task_ids):
    assert len(states) == len(terminals)
    trajectories = {}

    episode_num = 0
    for t in trange(len(terminals), desc="Segmenting"):
        if episode_num not in trajectories:
            trajectories[episode_num] = {
                "states": [],
                "actions": [],
                "rewards": [],
                "task_ids": [],
            }
        else:
            trajectories[episode_num]["states"].append(states[t])
            trajectories[episode_num]["actions"].append(actions[t])
            trajectories[episode_num]["rewards"].append(rewards[t])
            trajectories[episode_num]["task_ids"].append(task_ids[t])

        if terminals[t].item():
            # next episode
            episode_num = episode_num + 1

    trajectories_lens = [len(v["states"]) for k, v in trajectories.items()]

    for t in trajectories:
        trajectories[t]["states"] = np.stack(trajectories[t]["states"], axis=0)
        trajectories[t]["actions"] = np.stack(trajectories[t]["actions"], axis=0)
        trajectories[t]["rewards"] = np.stack(trajectories[t]["rewards"], axis=0)
        trajectories[t]["task_ids"] = np.stack(trajectories[t]["task_ids"], axis=0)

    return trajectories, trajectories_lens


if __name__ == "__main__":
    path = "/home/changyeon/CaDM_MerPO/data/walker-rand-param"
    data = offline_dataset(path, 50, range(5), range(5, 10))

    print(data["real_dones"].sum())
