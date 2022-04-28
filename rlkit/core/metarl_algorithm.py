import abc
from collections import OrderedDict
import time
import os
import glob
import gtimer as gt
import numpy as np
import torch

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler, OfflineInPlacePathSampler
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.sac.task import TASKAgent, EnsembleTransition


class OfflineMetaRLAlgorithm_ours(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            nets,
            train_tasks,
            eval_tasks,
            goal_radius,
            eval_deterministic=True,
            render=False,
            render_eval_paths=False,
            plotter=None,
            **kwargs
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval
        :param goal_radius: reward threshold for defining sparse rewards

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.env = env
        self.agent = nets[0].to(ptu.device)
        self.nets = nets
        self.qf1 = nets[1].to(ptu.device)
        self.qf2 = nets[2].to(ptu.device)
        self.policy = nets[3].to(ptu.device)
        self.train_tasks = train_tasks
        print("train tasks", self.train_tasks)
        self.eval_tasks = eval_tasks
        print("eval tasks", self.eval_tasks)
        self.goal_radius = goal_radius
        self.tasks_pool = []
        self.ad_qf_lr = kwargs['ad_qf_lr']

        self.meta_batch = kwargs['meta_batch']
        self.batch_size = kwargs['batch_size']
        self.num_iterations = kwargs['num_iterations']
        self.num_train_steps_per_itr = kwargs['num_train_steps_per_itr']
        self.max_path_length = kwargs['max_path_length']
        self.discount = kwargs['discount']
        self.replay_buffer_size = kwargs['replay_buffer_size']
        self.reward_scale = kwargs['reward_scale']
        self.use_automatic_beta_tuning = kwargs['use_automatic_beta_tuning']
        self.lagrange_thresh = kwargs['lagrange_thresh']
        self.num_test_steps_per_itr = kwargs['num_test_steps_per_itr']

        self.num_initial_steps = kwargs['num_initial_steps']
        self.num_tasks_sample = kwargs['num_tasks_sample']
        self.num_evals = kwargs['num_evals']
        self.num_steps_per_eval = kwargs['num_steps_per_eval']
        self.update_post_train = kwargs['update_post_train']
        self.num_steps_prior = kwargs['num_steps_prior']
        self.num_steps_posterior = kwargs['num_steps_posterior']
        self.num_extra_rl_steps_posterior = kwargs['num_extra_rl_steps_posterior']
        self.embedding_batch_size = kwargs['embedding_batch_size']
        self.embedding_mini_batch_size = kwargs['embedding_mini_batch_size']

        self.num_exp_traj_eval = kwargs['num_exp_traj_eval']
        self.save_replay_buffer = kwargs['save_replay_buffer']
        self.save_algorithm = kwargs['save_algorithm']
        self.save_environment = kwargs['save_environment']
        self.dump_eval_paths = kwargs['dump_eval_paths']
        self.data_dir = kwargs['data_dir']
        self.train_epoch = kwargs['train_epoch']
        self.eval_epoch = kwargs['eval_epoch']
        self.sample = kwargs['sample']
        self.n_trj = kwargs['n_trj']
        self.allow_eval = kwargs['allow_eval']
        self.mb_replace = kwargs['mb_replace']

        self.explore=kwargs['explore']
        self.net_size=300
        self.obs_shape = kwargs['obs_shape']
        self.action_shape = kwargs['action_shape']
        self.hidden_layer_size = kwargs['hidden_layer_size']
        self.transition_layers = kwargs['transition_layers']
        self.transition_init_num = kwargs['transition_init_num']
        self.device = kwargs['device']
        self.transition_lr = kwargs['transition_lr']
        self.use_automatic_entropy_tuning = kwargs["use_automatic_entropy_tuning"]
        self.target_entropy = kwargs["target_entropy"]
        self.task_policy_lr = kwargs["task_policy_lr"]
        self.lagrange_thresh = kwargs["lagrange_thresh"]
        self.qf_lr = kwargs["qf_lr"]
        self.buffer_size = kwargs['buffer_size']
        self.steps_per_inner_loop = kwargs['train_inner_step']
        self.real_data_ratio = kwargs['real_data_ratio']
        self.alpha_p_lr=kwargs['alpha_p_lr']

        self.use_automatic_beta_tuning = kwargs['use_automatic_beta_tuning']
        self.beta_init = kwargs['beta_init']
        self.test_only = kwargs['test_only']
        self.meta_test_only = kwargs['meta_test_only']

        self.checkpoint_paths_0 = kwargs['checkpoint_paths_0']
        self.checkpoint_paths_1 = kwargs['checkpoint_paths_1']
        self.finetune_model = kwargs['finetune_model']

        self.eval_deterministic = eval_deterministic
        self.render = render
        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.plotter = plotter

        self.train_buffer = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.train_tasks, self.goal_radius)
        self.eval_buffer = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.eval_tasks, self.goal_radius)
        self.replay_buffer = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.train_tasks, self.goal_radius)
        self.enc_replay_buffer = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.train_tasks, self.goal_radius)

        self.offline_sampler = OfflineInPlacePathSampler(env=env, policy=self.agent.meta_policy, max_path_length=self.max_path_length)

        self.sampler = InPlacePathSampler(env=env, policy=self.agent.meta_policy, max_path_length=self.max_path_length)

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self.tasks_qv1 = 0
        self.tasks_qv2 = 0
        self.test_tasks_qv1 = 0
        self.test_tasks_qv2 = 0

        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

        self.init_buffer()

        for net in self.nets:
            net.to(ptu.device)

    def init_buffer(self):
        train_trj_paths = []
        eval_trj_paths = []
        if self.sample:
            for n in range(self.n_trj):
                if self.train_epoch is None:
                    train_trj_paths += glob.glob(
                        os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step*.npy" % (n)))
                else:
                    for i in range(100):
                        train_trj_paths += glob.glob(
                            os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step%d.npy" % (n, i*self.train_epoch)))
                if self.eval_epoch is None:
                    eval_trj_paths += glob.glob(
                        os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step*.npy" % (n)))
                else:
                    for i in range(100):
                        eval_trj_paths += glob.glob(
                            os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step%d.npy" % (n, i*self.eval_epoch)))
        else:
            if self.train_epoch is None:
                train_trj_paths = glob.glob(
                    os.path.join(self.data_dir, "goal_idx*", "trj_eval[0-%d]_step*.npy") % (self.n_trj))
            else:
                train_trj_paths = glob.glob(os.path.join(self.data_dir, "goal_idx*",
                                                         "trj_eval[0-%d]_step%d.npy" % (self.n_trj, self.train_epoch)))
            if self.eval_epoch is None:
                eval_trj_paths = glob.glob(
                    os.path.join(self.data_dir, "goal_idx*", "trj_eval[0-%d]_step*.npy") % (self.n_trj))
            else:
                eval_trj_paths = glob.glob(os.path.join(self.data_dir, "goal_idx*",
                                                        "trj_eval[0-%d]_step%d.npy" % (self.n_trj, self.test_epoch)))

        train_paths = [train_trj_path for train_trj_path in train_trj_paths if
                       int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.train_tasks]
        train_task_idxs = [int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) for train_trj_path in
                           train_trj_paths if
                           int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.train_tasks]
        eval_paths = [eval_trj_path for eval_trj_path in eval_trj_paths if
                      int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.eval_tasks]
        eval_task_idxs = [int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) for eval_trj_path in eval_trj_paths if
                          int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.eval_tasks]

        obs_train_lst = []
        action_train_lst = []
        reward_train_lst = []
        next_obs_train_lst = []
        terminal_train_lst = []
        task_train_lst = []
        obs_eval_lst = []
        action_eval_lst = []
        reward_eval_lst = []
        next_obs_eval_lst = []
        terminal_eval_lst = []
        task_eval_lst = []

        for train_path, train_task_idx in zip(train_paths, train_task_idxs):
            trj_npy = np.load(train_path, allow_pickle=True)
            obs_train_lst += list(trj_npy[:, 0])
            action_train_lst += list(trj_npy[:, 1])
            reward_train_lst += list(trj_npy[:, 2])
            next_obs_train_lst += list(trj_npy[:, 3])
            terminal = [0 for _ in range(trj_npy.shape[0])]
            terminal[-1] = 1
            terminal_train_lst += terminal
            task_train = [train_task_idx for _ in range(trj_npy.shape[0])]
            task_train_lst += task_train

        for eval_path, eval_task_idx in zip(eval_paths, eval_task_idxs):
            trj_npy = np.load(eval_path, allow_pickle=True)
            obs_eval_lst += list(trj_npy[:, 0])
            action_eval_lst += list(trj_npy[:, 1])
            reward_eval_lst += list(trj_npy[:, 2])
            next_obs_eval_lst += list(trj_npy[:, 3])
            terminal = [0 for _ in range(trj_npy.shape[0])]
            terminal[-1] = 1
            terminal_eval_lst += terminal
            task_eval = [eval_task_idx for _ in range(trj_npy.shape[0])]
            task_eval_lst += task_eval

        for i, (
                task_train,
                obs,
                action,
                reward,
                next_obs,
                terminal,
        ) in enumerate(zip(
            task_train_lst,
            obs_train_lst,
            action_train_lst,
            reward_train_lst,
            next_obs_train_lst,
            terminal_train_lst,
        )):
            self.train_buffer.add_sample(
                task_train,
                obs,
                action,
                reward,
                terminal,
                next_obs,
                **{'env_info': {}},
            )

        for i, (
                task_eval,
                obs,
                action,
                reward,
                next_obs,
                terminal,
        ) in enumerate(zip(
            task_eval_lst,
            obs_eval_lst,
            action_eval_lst,
            reward_eval_lst,
            next_obs_eval_lst,
            terminal_eval_lst,
        )):
            self.eval_buffer.add_sample(
                task_eval,
                obs,
                action,
                reward,
                terminal,
                next_obs,
                **{'env_info': {}},
            )

        task_para_list = [
            self.obs_shape,
            self.action_shape,
            self.hidden_layer_size,
            self.transition_layers,
            self.transition_init_num,
            self.device,
            self.transition_lr,
            self.use_automatic_entropy_tuning,
            self.target_entropy,
            self.task_policy_lr,
            self.lagrange_thresh,
            self.qf_lr,
            self.buffer_size,
            self.goal_radius,
            self.net_size,
            self.use_automatic_beta_tuning,
            self.beta_init,
            self.alpha_p_lr,
            self.ad_qf_lr,
        ]

        inner_dnns, test_dnns = self.load_model(self.obs_shape, self.action_shape, self.hidden_layer_size, self.transition_layers)

        for index in range(len(self.train_tasks)):

            train_task = TASKAgent(index, self.nets, task_para_list, inner_dnns, train_test=True)
            self.tasks_pool.append(train_task)

        for index in range(len(self.eval_tasks)):
            eval_task = TASKAgent(index,
                                  self.nets,
                                  task_para_list,
                                  test_dnns,
                                  train_test=False
                                  )
            self.tasks_pool.append(eval_task)

    def _try_to_eval(self, epoch):

        if self._can_evaluate():
            self.evaluate(epoch)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys
            logger.record_tabular("task_avg_q1", self.tasks_qv1)
            logger.record_tabular("task_avg_q2", self.tasks_qv2)
            logger.record_tabular("test_task_avg_q1", self.test_tasks_qv1)
            logger.record_tabular("test_task_avg_q2", self.test_tasks_qv2)
            logger.record_tabular("Number of train steps total", self._n_train_steps_total)
            logger.record_tabular("Number of env steps total", self._n_env_steps_total)
            logger.record_tabular("Number of rollouts total", self._n_rollouts_total)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def load_model(self, observation_dims, action_dims, hidden_feature_number, hidden_layer_number):
        dimensions = [observation_dims, action_dims]
        hidden_features = hidden_feature_number
        hidden_layers = hidden_layer_number
        checkpoint_train = torch.load(self.checkpoint_paths_0, map_location=ptu.device)
        checkpoint_test = torch.load(self.checkpoint_paths_1, map_location=ptu.device)

        outer_dnn = EnsembleTransition(dimensions[0], dimensions[1], hidden_features, hidden_layers).to(
                                   device=ptu.device)
        if self.finetune_model != "scratch":
            outer_dnn.load_state_dict(checkpoint_train["meta_model"])

        inner_dnns = []
        for i in range(len(self.train_tasks)):
            inner_dnns.append(EnsembleTransition(dimensions[0], dimensions[1], hidden_features, hidden_layers).to(
                                             device=ptu.device))
            if self.finetune_model == "outer":
                inner_dnns[i].load_state_dict(checkpoint_train['meta_model'])
            elif self.finetune_model == "inner":
                inner_dnns[i].load_state_dict(checkpoint_train['training_task_models'][i])
        test_dnns = []

        for i in range(len(self.eval_tasks)):
            test_dnns.append(EnsembleTransition(dimensions[0], dimensions[1], hidden_features, hidden_layers).to(
                                            device=ptu.device))
            if self.finetune_model == "outer":
                test_dnns[i].load_state_dict(checkpoint_train['meta_model'])
            elif self.finetune_model == "inner":
                test_dnns[i].load_state_dict(checkpoint_test['testing_trained_task_models'][i])
        return inner_dnns, test_dnns

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation, )

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.agent.meta_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def _do_eval(self, indices, epoch, buffer):
        final_returns = []
        online_returns = []

        test_tasks_qf1 = []
        test_tasks_qf2 = []
        for idx in indices:
            test=False
            all_rets = []
            idx_list = [idx]

            if self.test_only is False:
                if (set(indices).issubset(set(self.eval_tasks))):

                    test_task_qf1, test_task_qf2 = self._do_adaptation(idx_list, self.eval_buffer)
                    test_tasks_qf1.append(test_task_qf1)
                    test_tasks_qf2.append(test_task_qf2)
                    test = True
            else:
                test = True

            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r, buffer, test)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([a[-1] for a in all_rets]))

            n = min([len(a) for a in all_rets])

            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0)
            online_returns.append(all_rets)

        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        self.test_tasks_qv1 = np.mean(test_tasks_qf1)
        self.test_tasks_qv2 = np.mean(test_tasks_qf2)
        return final_returns, online_returns

    def test(self, log_dir, end_point=-1):
        assert os.path.exists(log_dir)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        for it_ in gt.timed_for(range(self.num_iterations), save_itrs=True):
            self._start_epoch(it_)

            if it_ == 0:

                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.collect_data(self.num_initial_steps, 1, np.inf, buffer=self.train_buffer) #from the buffer

            for i in range(self.num_tasks_sample):
                idx = np.random.choice(self.train_tasks, 1)[0]
                self.task_idx = idx
                self.env.reset_task(idx)
                self.enc_replay_buffer.task_buffers[idx].clear()


                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf, buffer=self.train_buffer)

                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train, buffer=self.train_buffer)

                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train,
                                      buffer=self.train_buffer,
                                      add_to_enc_buffer=False)


            for train_step in range(self.num_train_steps_per_itr):
                self._n_train_steps_total += 1

            gt.stamp('train')

            self.training_mode(False)

            if it_ % 5 == 0 and it_ > end_point:
                status = self.load_epoch_model(it_, log_dir)
                if status:
                    self._try_to_eval(it_)
            gt.stamp('eval')
            self._end_epoch()

    def train(self):
        '''
        meta-training loop
        '''


        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        if self.test_only is False:

            for it_ in gt.timed_for(range(self.num_iterations), save_itrs=True):
                self._start_epoch(it_)
                self.training_mode(True)
                if it_ == 0:

                    for idx in self.train_tasks:
                        self.task_idx = idx
                        self.env.reset_task(idx)
                        self.collect_data(self.num_initial_steps, 1, np.inf, buffer=self.train_buffer)


                for i in range(self.num_tasks_sample):
                    idx = np.random.choice(self.train_tasks, 1)[0]
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.enc_replay_buffer.task_buffers[idx].clear()

                indices_lst = []


                for train_step in range(self.num_train_steps_per_itr):

                    indices = np.random.choice(self.train_tasks, self.meta_batch, replace=self.mb_replace) #mod

                    tasks_qv1, tasks_qv2 = self._do_training(indices, it_, test=False)
                    indices_lst.append(indices)
                    self._n_train_steps_total += 1


                self.tasks_qv1 = np.mean(tasks_qv1)
                self.tasks_qv2 = np.mean(tasks_qv2)


                gt.stamp('train')
                self.training_mode(True)

                params = self.get_epoch_snapshot(it_)
                logger.save_itr_params(it_, params)

                if self.allow_eval:
                    logger.save_extra_data(self.get_extra_data_to_save(it_))

                    self._try_to_eval(it_)
                    gt.stamp('eval')
                self._end_epoch()

        else:
            print("load meta policy")
            path = "./output/walker-rand-params/2021_06_30_11_11_11_seed1"
            epoch = 480
            self.agent.meta_policy.load_state_dict(torch.load(os.path.join(path, 'policy_itr_{}.pth'.format(epoch))))
            if self.meta_test_only:
                print("load policy")
                path = "./output/walker-rand-params/2021_06_30_11_11_11_seed1"
                epoch = 480
                for task_i in self.eval_tasks:

                    self.tasks_pool[task_i].log_alpha=torch.zeros(1, requires_grad=True, device=ptu.device)
                    if self.use_automatic_beta_tuning:
                        self.tasks_pool[task_i].beta=torch.tensor(self.beta_init, device=ptu.device, requires_grad=True)
                    if self.lagrange_thresh>0:
                        self.tasks_pool[task_i].log_alpha_prime=torch.zeros(1, requires_grad=True, device=ptu.device)

            else:
                print("load policy")
                path = "./output/walker-rand-params/2021_06_30_11_11_11_seed1"
                epoch = 480
                for task_i in self.eval_tasks:

                    self.tasks_pool[task_i].log_alpha=torch.zeros(1, requires_grad=True, device=ptu.device)

                    if self.use_automatic_beta_tuning:
                        self.tasks_pool[task_i].beta=torch.tensor(self.beta_init, device=ptu.device, requires_grad=True)
                    if self.lagrange_thresh>0:
                        self.tasks_pool[task_i].log_alpha_prime=torch.zeros(1, requires_grad=True, device=ptu.device)

            for it_ in gt.timed_for(range(self.num_iterations), save_itrs=True):

                self._start_epoch(it_)
                indices_lst=[]
                for train_step in range(self.num_test_steps_per_itr):

                    tasks_qv1, tasks_qv2 = self._do_training(indices, it_, test=True)
                    indices_lst.append(indices)
                    self._n_train_steps_total += 1


                self.tasks_qv1 = np.mean(tasks_qv1)
                self.tasks_qv2 = np.mean(tasks_qv2)
                gt.stamp('train')
                self.training_mode(True)

                if self.allow_eval:
                    self._try_to_eval(it_)
                    gt.stamp('eval')

                self._end_epoch()












    def data_dict(self, indices):
        data_dict = {}
        data_dict['task_idx'] = indices
        return data_dict

    def evaluate(self, epoch):

        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        if len(self.train_tasks) == 2 and len(self.eval_tasks) == 2: # train tasks same with test tasks in this case
            indices = self.train_tasks
            eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))

            train_returns = []
            for idx in indices:
                self.task_idx = idx
                self.env.reset_task(idx)
                paths = []

                for _ in range(self.num_steps_per_eval // self.max_path_length):
                    p, _ = self.offline_sampler.obtain_samples(buffer=self.train_buffer,
                                                               policy=self.tasks_pool[idx].task_policy,
                                                               deterministic=self.eval_deterministic,
                                                               max_samples=self.max_path_length,
                                                               accum_context=False,
                                                               max_trajs=1,
                                                               resample=np.inf)
                    paths += p

                if self.sparse_rewards:
                    for p in paths:
                        sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                        p['rewards'] = sparse_rewards

                train_returns.append(eval_util.get_average_returns(paths))

            train_final_returns, train_online_returns = self._do_eval(self.train_tasks, epoch, buffer=self.train_buffer)

            eval_util.dprint('train online returns')
            eval_util.dprint(train_online_returns)


            eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
            test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch, buffer=self.eval_buffer)
            eval_util.dprint('test online returns')
            eval_util.dprint(test_online_returns)



            if hasattr(self.env, "log_diagnostics"):
                self.env.log_diagnostics(paths, prefix=None)

            avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
            avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
            print("avg_online_test_return: ", avg_test_online_return)
            for i in indices:
                self.eval_statistics[f'AverageTrainReturn_train_task{i}'] = train_returns[i]
                self.eval_statistics[f'AverageReturn_all_train_task{i}'] = train_final_returns[i]
                self.eval_statistics[f'AverageReturn_all_test_tasks{i}'] = test_final_returns[i]

            self.eval_statistics[f'AverageReturn_all_test_tasks'] = np.mean(test_final_returns)


        else:
            indices = np.random.choice(self.train_tasks, 8) #mod

            train_returns = []
            for idx in indices:
                self.task_idx = idx
                self.env.reset_task(idx)
                paths = []
                for _ in range(self.num_steps_per_eval // self.max_path_length):

                    p, _ = self.offline_sampler.obtain_samples(buffer=self.train_buffer,
                                                               policy=self.tasks_pool[idx].task_policy,
                                                               deterministic=self.eval_deterministic,
                                                               max_samples=self.max_path_length,
                                                               accum_context=False,
                                                               max_trajs=1,
                                                               resample=np.inf)
                    paths += p

                if self.sparse_rewards:
                    for p in paths:
                        sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                        p['rewards'] = sparse_rewards

                train_returns.append(eval_util.get_average_returns(paths))
            train_returns = np.mean(train_returns)

            train_final_returns, train_online_returns = self._do_eval(indices, epoch, buffer=self.train_buffer)
            eval_util.dprint('train online returns')
            eval_util.dprint(train_online_returns)


            test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch, buffer=self.eval_buffer)
            eval_util.dprint('test online returns')
            eval_util.dprint(test_online_returns)



            if hasattr(self.env, "log_diagnostics"):
                self.env.log_diagnostics(paths, prefix=None)

            avg_train_return = np.mean(train_final_returns)
            avg_test_return = np.mean(test_final_returns)
            avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
            avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)

            #self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
            self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
            self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
            self.eval_statistics['AverageReturn_online_test_tasks'] = np.mean(avg_test_online_return)

            self.loss['train_returns'] = train_returns
            self.loss['avg_test_return'] = avg_test_return
            self.loss['avg_train_online_return'] = np.mean(avg_train_online_return)
            self.loss['avg_test_online_return'] = np.mean(avg_test_online_return)



        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()


    def collect_paths(self, idx, epoch, run, buffer,test):
        self.task_idx = idx
        self.env.reset_task(idx)

        paths = []
        num_transitions = 0
        while num_transitions < self.num_steps_per_eval:
            path, num = self.offline_sampler.obtain_samples(
                buffer=buffer,
                policy=self.tasks_pool[idx].task_policy,
                deterministic=self.eval_deterministic,
                max_samples=self.num_steps_per_eval - num_transitions,
                max_trajs=1,
                accum_context=False,
                rollout=True)
            paths += path
            num_transitions += num
            break

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal



        return paths


    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, buffer, add_to_enc_buffer=False):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.offline_sampler.obtain_samples(buffer=buffer,
                                                                   policy=self.agent.meta_policy,
                                                                   max_samples=num_samples - num_transitions,
                                                                   max_trajs=update_posterior_rate,
                                                                   accum_context=False,
                                                                   resample=resample_z_rate,
                                                                   rollout=False)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)

        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def load_transition(self, transition_path):
        transition = torch.load(transition_path)
        return transition

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

    @abc.abstractmethod
    def _do_adaptation(self):
        """
        Perform policy learning with meta-policy for new tasks.
        :return:
        """
        pass
