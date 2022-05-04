import numpy as np
from rand_param_envs.base import RandomEnv
from gym import utils
import torch

class HalfCheetahRandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0):
        RandomEnv.__init__(self, log_scale_limit, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(a).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        ob = np.concatenate([np.array(reward).reshape(1), ob])
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        reward = 0.
        ob = self._get_obs()
        return np.concatenate([np.array(reward).reshape(1), ob])

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def torch_reward_fn(self):
        def _thunk(obs, act, next_obs):
            return obs[:,0]
            # if isinstance(act, torch.Tensor):
            #     act = act
            # posbefore = self.data.qpos[0]
            # self.do_simulation(act, self.frame_skip)
            # posafter, height, ang = self.data.qpos[0:3]
            # alive_bonus = 1.0
            # reward = ((posafter - posbefore / self.dt))
            # reward += alive_bonus
            # reward -= 1e-3 * np.square(act).sum()
            # return reward
        return _thunk

    def torch_done_fn(self):
        def _thunk(next_obs):
            done = False
            done = torch.tensor(done)
            return done
        return _thunk

if __name__ == "__main__":

    env = HalfCheetahRandParamsEnv()
    tasks = env.sample_tasks(40)
    while True:
        env.reset()
        env.set_task(np.random.choice(tasks))
        print(env.model.body_mass)
        for _ in range(100):
            #env.render()
            env.step(env.action_space.sample())  # take a random action