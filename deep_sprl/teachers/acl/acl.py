import numpy as np
from deep_sprl.teachers.acl.exp3s import Exp3S
from deep_sprl.teachers.abstract_teacher import AbstractTeacher, BaseWrapper


class ACL(AbstractTeacher):

    def __init__(self, n_contexts, eta, eps=0.2, norm_hist_len=1000):
        self.bandit = Exp3S(n_contexts, eta, eps=eps, norm_hist_len=norm_hist_len)
        self.last_rewards = [None] * n_contexts

    def update(self, i, r):
        if self.last_rewards[i] is None:
            self.last_rewards[i] = r
            self.bandit.update(i, 0.)
        else:
            progress = np.abs(r - self.last_rewards[i])
            self.last_rewards[i] = r
            self.bandit.update(i, progress)

    def sample(self):
        return self.bandit.sample()

    def save(self, path):
        pass

    def load(self, path):
        pass


class ACLWrapper(BaseWrapper):

    def __init__(self, env, acl, discount_factor, context_visible, context_post_processing=None):
        BaseWrapper.__init__(self, env, acl, discount_factor, context_visible,
                             context_post_processing=context_post_processing)

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward):
        self.teacher.update(cur_context, discounted_reward)
