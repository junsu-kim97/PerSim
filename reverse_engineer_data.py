import glob
import os
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    data_dir = '/home/junsu/workspace/faster-trajectory-transformer/data/new-walker-rand-params'
    trj_paths = []
    tasks = range(0, 24)
    idxs = range(0, 50)

    for n in tasks:
        for i in idxs:
            trj_paths += glob.glob(os.path.join(data_dir, "goal_idx%d" % (n),  "trj_evalsample%d" % (i) + "_step*.npy"))

    paths = [
        trj_path
        for trj_path in trj_paths
        if int(trj_path.split("/")[-2].split("goal_idx")[-1]) in tasks
    ]

    dt = 0.008
    for path in tqdm(paths):
        dataset = np.load(path, allow_pickle=True)
        for idx, (obs, action, reward, next_obs) in enumerate(dataset):
            print("loop")
    print("done")
