import numpy as np
from rand_param_envs.base import RandomEnv
from rand_param_envs.gym import utils
import torch

class Walker2DRandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0):
        RandomEnv.__init__(self, log_scale_limit, 'walker2d.xml', 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        ob = np.concatenate([np.array(reward).reshape(1), ob])
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        reward = 1.0
        ob = self._get_obs()
        return np.concatenate([np.array(reward).reshape(1), ob])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

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
            posafter, height, ang = self.data.qpos[0:3]
            done = not (height > 0.8 and height < 2.0 and
                        ang > -1.0 and ang < 1.0)
            done = torch.tensor(done)
            return done
        return _thunk

if __name__ == "__main__":

    env = Walker2DRandParamsEnv()
    tasks = env.sample_tasks(40)
    while True:
        env.reset()
        env.set_task(np.random.choice(tasks))
        print(env.model.body_mass)
        for _ in range(100):
            env.render()
            env.step(env.action_space.sample())  # take a random action

