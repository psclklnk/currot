import math
import torch
import numpy as np
from deep_sprl.teachers.vds.replay_buffer import ReplayBuffer
from deep_sprl.teachers.abstract_teacher import AbstractTeacher, BaseWrapper


class VDS(AbstractTeacher):

    def __init__(self, context_lb, context_ub, gamma, n_q, n_samples=1000, net_arch=None, q_train_config=None,
                 is_discrete=False):
        if net_arch is None:
            net_arch = {"layers": [128, 128, 128], "act_func": torch.nn.Tanh()}
        self.net_arch = net_arch

        if q_train_config is None:
            q_train_config = {"replay_size": 20000, "lr": 1e-4, "n_epochs": 10, "batches_per_epoch": 50,
                              "steps_per_update": 4096}
        self.q_train_config = q_train_config
        self.n_q = n_q

        self.gamma = gamma
        self.context_lb = context_lb
        self.context_ub = context_ub
        self.is_discrete = is_discrete
        self.n_samples = n_samples
        self.next_update = q_train_config["steps_per_update"]

        # Will be create in the init method
        self.replay_buffer = None
        self.learner = None
        self.state_provider = None
        self.qs = None
        self.optimizer = None

        # Will be create when sampling
        self.contexts = None
        self.likelihoods = None

    def initialize_teacher(self, env, learner, state_provider):
        obs_shape = env.observation_space.shape
        action_dim = env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(self.q_train_config["replay_size"], obs_shape, action_dim,
                                          handle_timeout_termination=False)
        self.learner = learner
        self.state_provider = state_provider

        self.qs = EnsembleQFunction(**self.net_arch, input_dim=obs_shape[0] + action_dim, k=self.n_q)
        self.optimizer = torch.optim.Adam(self.qs.parameters(), lr=self.q_train_config["lr"])

    def update(self, count):
        if count >= self.next_update:
            print("Update Q-Ensemble")
            # Train the Q-Function
            for _ in range(self.q_train_config["n_epochs"]):
                batch_size = self.replay_buffer.size() // self.q_train_config["batches_per_epoch"] + 1
                for i in range(self.q_train_config["batches_per_epoch"]):
                    obs, acts, next_obs, dones, rewards = self.replay_buffer.sample((self.n_q, batch_size))
                    next_actions = torch.from_numpy(
                        self.learner.get_action(next_obs.detach().numpy()))
                    with torch.no_grad():
                        next_q_values = self.qs(torch.cat((next_obs, next_actions), axis=-1))
                    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
                    current_q_values = self.qs(torch.cat((obs, acts), axis=-1))
                    loss = torch.sum(torch.nn.functional.mse_loss(current_q_values, target_q_values))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            print("Finished Updating")

            # Update the sample buffer (we simply sample 2000 points and pre-compute the priority)
            if self.is_discrete:
                self.contexts = np.arange(self.context_lb, self.context_ub)
            else:
                self.contexts = np.random.uniform(self.context_lb, self.context_ub,
                                                  size=(self.n_samples, len(self.context_lb)))

            states = self.state_provider(self.contexts)
            actions = self.learner.get_action(states)
            q_inputs = np.concatenate((states, actions), axis=-1)
            disagreements = np.std(np.squeeze(self.qs(torch.from_numpy(q_inputs).type(torch.float32)).detach().numpy()),
                                   axis=0)
            self.likelihoods = disagreements / np.sum(disagreements)

            # Increase the update counter
            self.next_update += self.q_train_config["steps_per_update"]

    def sample(self):
        # Sample uniformly over the context space
        if self.qs is None or self.contexts is None:
            if self.is_discrete:
                return np.array(np.random.randint(self.context_lb, self.context_ub))
            else:
                return np.random.uniform(self.context_lb, self.context_ub)
        else:
            return self.contexts[np.random.choice(self.contexts.shape[0], p=self.likelihoods), ...]

    def save(self, path):
        pass

    def load(self, path):
        pass


class EnsembleQFunction(torch.nn.Module):

    def __init__(self, input_dim, layers, act_func, k):
        super().__init__()
        layers_ext = [input_dim] + layers + [1]
        torch_layers = []
        for i in range(len(layers_ext) - 1):
            torch_layers.append(EnsembleLinear(layers_ext[i], layers_ext[i + 1], k, bias=True))
        self.layers = torch.nn.ModuleList(torch_layers)
        self.act_fun = act_func

    def __call__(self, x):
        h = x
        for l in self.layers[:-1]:
            h = self.act_fun(l(h))

        return self.layers[-1](h)


class EnsembleLinear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features', 'k']
    in_features: int
    out_features: int
    k: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, k: int, bias: bool = True) -> None:
        super(EnsembleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.weight = torch.nn.Parameter(torch.Tensor(k, out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(k, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.k):
            torch.nn.init.kaiming_uniform_(self.weight[i, ...], a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0, ...])
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # In this case we compute the predictions of the ensembles for the same data
        if len(input.shape) == 2:
            x = torch.einsum("kij,nj->kni", self.weight, input)
        # Here we compute the predictions of the ensembles for the data independently
        elif len(input.shape) == 3:
            x = torch.einsum("kij,knj->kni", self.weight, input)
        else:
            raise RuntimeError("Ensemble only supports predictions with 2- or 3D input")

        if self.bias is not None:
            return x + self.bias[:, None, :]
        else:
            return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, k={}, bias={}'.format(
            self.in_features, self.out_features, self.k, self.bias is not None
        )


class VDSWrapper(BaseWrapper):

    def __init__(self, env, vds, discount_factor, context_visible, context_post_processing=None):
        BaseWrapper.__init__(self, env, vds, discount_factor, context_visible,
                             context_post_processing=context_post_processing)
        self.last_obs = None
        self.step_count = 0

    def reset(self):
        self.cur_context = self.teacher.sample()
        if self.context_post_processing is None:
            self.processed_context = self.cur_context.copy()
        else:
            self.processed_context = self.context_post_processing(self.cur_context).copy()
        self.env.unwrapped.context = self.processed_context.copy()
        obs = self.env.reset()

        if self.context_visible:
            obs = np.concatenate((obs, self.processed_context))

        self.last_obs = obs.copy()
        self.cur_initial_state = obs.copy()
        return obs

    def step(self, action):
        step = self.env.step(action)
        if self.context_visible:
            step = np.concatenate((step[0], self.processed_context)), step[1], step[2], step[3]
        self.teacher.replay_buffer.add(self.last_obs, step[0].copy(), action, step[1], step[2], [])
        self.last_obs = step[0].copy()
        self.step_count += 1
        self.update(step)
        return step

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward):
        # We currently rely on the learner being set on the environment after its creation
        self.teacher.update(self.step_count)
