import gym
import numpy as np
import multiprocessing
from abc import ABC, abstractmethod
from deep_sprl.teachers.util import Buffer
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, CloudpickleWrapper
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _flatten_obs
from stable_baselines3.common.running_mean_std import RunningMeanStd


class AbstractTeacher(ABC):

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass


class BaseWrapper(gym.Env):

    def __init__(self, env, teacher, discount_factor, context_visible, reward_from_info=False,
                 context_post_processing=None):
        gym.Env.__init__(self)
        self.stats_buffer = Buffer(3, 10000, True)
        self.context_trace_buffer = Buffer(3, 10000, True)

        self.env = env
        self.teacher = teacher
        self.discount_factor = discount_factor

        if context_visible:
            context = self.teacher.sample()
            if context_post_processing is not None:
                context = context_post_processing(context)

            low_ext = np.concatenate((self.env.observation_space.low, -np.inf * np.ones_like(context)))
            high_ext = np.concatenate((self.env.observation_space.high, np.inf * np.ones_like(context)))
            self.observation_space = gym.spaces.Box(low=low_ext, high=high_ext)
        else:
            self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        if hasattr(self.env, "reward_range"):
            self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

        self.undiscounted_reward = 0.
        self.discounted_reward = 0.
        self.cur_disc = 1.
        self.step_length = 0.

        self.context_visible = context_visible
        self.cur_context = None
        self.processed_context = None
        self.cur_initial_state = None

        self.reward_from_info = reward_from_info
        self.context_post_processing = context_post_processing

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward):
        pass

    def step(self, action):
        step = self.env.step(action)
        if self.context_visible:
            step = np.concatenate((step[0], self.processed_context)), step[1], step[2], step[3]
        self.update(step)
        return step

    def reset(self):
        self.cur_context = self.teacher.sample()
        if self.context_post_processing is None:
            self.processed_context = self.cur_context.copy()
        else:
            self.processed_context = self.context_post_processing(self.cur_context.copy())
        self.env.unwrapped.context = self.processed_context.copy()
        obs = self.env.reset()

        if self.context_visible:
            obs = np.concatenate((obs, self.processed_context))

        self.cur_initial_state = obs.copy()
        return obs

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def update(self, step):
        reward = step[3]["reward"] if self.reward_from_info else step[1]
        self.undiscounted_reward += reward
        self.discounted_reward += self.cur_disc * reward
        self.cur_disc *= self.discount_factor
        self.step_length += 1.

        if step[2]:
            self.done_callback(step, self.cur_initial_state.copy(), self.cur_context, self.discounted_reward,
                               self.undiscounted_reward)

            self.stats_buffer.update_buffer((self.undiscounted_reward, self.discounted_reward, self.step_length))
            self.context_trace_buffer.update_buffer((self.undiscounted_reward, self.discounted_reward,
                                                     self.processed_context.copy()))
            self.undiscounted_reward = 0.
            self.discounted_reward = 0.
            self.cur_disc = 1.
            self.step_length = 0.

            self.cur_context = None
            self.processed_context = None
            self.cur_initial_state = None

    def get_statistics(self):
        if len(self.stats_buffer) == 0:
            return 0., 0., 0
        else:
            rewards, disc_rewards, steps = self.stats_buffer.read_buffer()
            mean_reward = np.mean(rewards)
            mean_disc_reward = np.mean(disc_rewards)
            mean_step_length = np.mean(steps)

            return mean_reward, mean_disc_reward, mean_step_length

    def get_encountered_contexts(self):
        return self.context_trace_buffer.read_buffer()


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                ret = env.step(data)
                remote.send(ret)
            elif cmd == 'reset':
                env.unwrapped.context = data
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == 'seed':
                remote.send(env.seed(data))
            else:
                raise NotImplementedError("Unknown command: %s" % cmd)
        except EOFError:
            break


class BaseVecEnvWrapper(SubprocVecEnv):

    def __init__(self, env_fns, teacher, discount_factor, context_visible, start_method=None, reward_from_info=False,
                 context_post_processing=None, normalize_rewards=False):
        ###########################################################
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        #############################################################

        self.stats_buffer = Buffer(3, 10000, True)
        self.context_trace_buffer = Buffer(3, 10000, True)
        self.teacher = teacher
        self.discount_factor = discount_factor

        if context_visible:
            context = self.teacher.sample()
            if context_post_processing is not None:
                context = context_post_processing(context)

            low_ext = np.concatenate((self.observation_space.low, -np.inf * np.ones_like(context)))
            high_ext = np.concatenate((self.observation_space.high, np.inf * np.ones_like(context)))
            self.observation_space = gym.spaces.Box(low=low_ext, high=high_ext)
        else:
            self.observation_space = self.observation_space
        self.action_space = self.action_space

        self.undiscounted_rewards = np.zeros(len(env_fns))
        self.discounted_rewards = np.zeros(len(env_fns))
        self.cur_discs = np.ones(len(env_fns))
        self.step_lengths = np.zeros(len(env_fns))

        self.context_visible = context_visible
        self.cur_contexts = [None] * len(env_fns)
        self.processed_contexts = [None] * len(env_fns)
        self.cur_initial_states = [None] * len(env_fns)

        self.reward_from_info = reward_from_info
        self.context_post_processing = context_post_processing

        if normalize_rewards:
            self.rets = np.zeros(len(env_fns))
            self.ret_rms = RunningMeanStd(shape=())
        else:
            self.ret_rms = None

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset_async(self, env_id, force=False):
        if self.cur_contexts[env_id] is not None and not force:
            raise RuntimeError("Environment has not terminated before being resetted")

        context = self.teacher.sample()
        self.cur_contexts[env_id] = context.copy()
        if self.context_post_processing is None:
            self.processed_contexts[env_id] = self.cur_contexts[env_id].copy()
        else:
            self.processed_contexts[env_id] = self.context_post_processing(self.cur_contexts[env_id].copy())
        self.remotes[env_id].send(('reset', self.processed_contexts[env_id].copy()))

    def reset_wait(self, env_id):
        obs = self.remotes[env_id].recv()
        if self.context_visible:
            obs = np.concatenate([obs, self.processed_contexts[env_id]], axis=0)
        self.cur_initial_states[env_id] = obs.copy()
        return obs

    def reset(self, force=False):
        for i in range(0, len(self.remotes)):
            # In this case we need to make sure that the buffers are cleared!
            if force:
                self.reset_data(i)
            self.reset_async(i, force=force)
        obs = [self.reset_wait(i) for i in range(0, len(self.remotes))]
        return _flatten_obs(obs, self.observation_space)

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        if self.context_visible:
            obs = [np.concatenate([ob, self.processed_contexts[i]], axis=0) for i, ob in enumerate(obs)]

        step = (obs, rewards, dones, infos)
        self.update(step)

        # Do the automatic resetting of the environments
        new_obs = []
        for i, (ob, done, info) in enumerate(zip(obs, dones, infos)):
            if done:
                info['terminal_observation'] = ob

                # Reset the environment
                self.reset_async(i)
                ob = self.reset_wait(i)

            new_obs.append(ob)

        if self.ret_rms is None:
            normalized_rewards = np.stack(rewards)
        else:
            normalized_rewards = np.clip(np.stack(rewards) / np.sqrt(self.ret_rms.var + 1e-4), -10., 10.)
        return _flatten_obs(new_obs, self.observation_space), normalized_rewards, np.stack(dones), infos

    def reset_data(self, env_id):
        self.undiscounted_rewards[env_id] = 0.
        self.discounted_rewards[env_id] = 0.
        self.cur_discs[env_id] = 1.
        self.step_lengths[env_id] = 0.

        self.cur_contexts[env_id] = None
        self.cur_initial_states[env_id] = None

    def update(self, step):
        rewards = step[3]["reward"] if self.reward_from_info else step[1]
        self.undiscounted_rewards += rewards
        self.discounted_rewards += self.cur_discs * rewards
        self.cur_discs *= self.discount_factor
        self.step_lengths += 1.
        if self.ret_rms is not None:
            self.rets = self.rets * self.discount_factor + step[1]
            self.ret_rms.update(self.rets)
            self.rets[step[2]] = 0.

        for i in range(0, len(self.processes)):
            if step[2][i]:
                cur_step = (step[0][i], step[1][i], step[2][i], step[3][i])
                self.done_callback(cur_step, self.cur_initial_states[i].copy(), self.cur_contexts[i].copy(),
                                   self.discounted_rewards[i], self.undiscounted_rewards[i])

                self.stats_buffer.update_buffer((self.undiscounted_rewards[i], self.discounted_rewards[i],
                                                 self.step_lengths[i]))
                self.context_trace_buffer.update_buffer((self.undiscounted_rewards[i], self.discounted_rewards[i],
                                                         self.cur_contexts[i].copy()))
                self.reset_data(i)

    def get_statistics(self):
        if len(self.stats_buffer) == 0:
            return 0., 0., 0
        else:
            rewards, disc_rewards, steps = self.stats_buffer.read_buffer()
            mean_reward = np.mean(rewards)
            mean_disc_reward = np.mean(disc_rewards)
            mean_step_length = np.mean(steps)

            return mean_reward, mean_disc_reward, mean_step_length

    def get_encountered_contexts(self):
        return self.context_trace_buffer.read_buffer()

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward):
        pass
