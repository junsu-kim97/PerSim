# PerSim: Data-Efficient Offline Reinforcement Learning  with Heterogeneous Agents via Personalized Simulators

This is the code accompaying the paper submission **PerSim: Data-Efficient Offline Reinforcement Learning  with Heterogeneous Agents via Personalized Simulators"** 


## Requirements

* `python >3.6`
* `Mujoco-py ` and its [prerequisites](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key). 
* python packages in `requirements.txt`

## Datasets

We provide the offline datasets we performed the experiments on. The datasets can be downloaded via running `data.sh` through:
	
	`bash data.sh`


## Running PerSim

To run PerSim, run the following script:

	`python3 run.py --env {env} --dataname {dataname} --r {rank}`

Choose env from {`mountainCar`, `cartPole`, `halfCheetah`}, and dataname from the available datasets in the `datasets` directory. e.g., `cartPole_pure_0.0_0`. Best values for r is 3,5,15 for mountainCar, cartPole, and halfCheetah respectively.

./scripts/train_our_data.sh 4 walker-rand-params walker-rand-params_ours-v3_0.0_0 15

./scripts/train_our_data_half_cheetah.sh 2 half-cheetah-rand-params half-cheetah-rand-params_ours-0503-v3-300-300_0.0_0 15 300 300

./scripts/train_our_data_debug_half_cheetah.sh 1 half-cheetah-rand-params cheetah_debug_0.0_0 15 300 1