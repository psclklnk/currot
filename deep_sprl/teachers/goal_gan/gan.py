import copy
import torch
import numpy as np
from functools import partial

# Here we use a function to avoid reuse of objects
DEFAULT_GAN_CONFIGS = lambda: {
    'batch_size': 64,
    'generator_output_activation': 'tanh',
    'generator_hidden_activation': 'relu',
    'discriminator_hidden_activation': 'leaky_relu',
    'generator_optimizer': torch.optim.RMSprop,
    'discriminator_optimizer': torch.optim.RMSprop,
    'generator_weight_initializer': torch.nn.init.xavier_uniform,  # tf.contrib.layers.xavier_initializer(),
    'discriminator_weight_initializer': torch.nn.init.xavier_uniform,  # tf.contrib.layers.xavier_initializer(),
    'print_iteration': 50,
    'supress_all_logging': False,
    'default_generator_iters': 1,  # It is highly recommend to not change these parameters
    'default_discriminator_iters': 1,
    'gan_type': 'lsgan',
    'wgan_gradient_penalty': 0.1,
}


def batch_feed_array(array, batch_size):
    data_size = array.shape[0]

    if data_size <= batch_size:
        while True:
            yield array
    else:
        start = 0
        while True:
            if start + batch_size < data_size:
                yield array[start:start + batch_size, ...]
            else:
                yield np.concatenate(
                    [array[start:data_size], array[0: start + batch_size - data_size]],
                    axis=0
                )
            start = (start + batch_size) % data_size


class FCGAN(object):
    def __init__(self, generator_output_size, discriminator_output_size, generator_layers, discriminator_layers,
                 noise_size, configs=None):

        self.generator_output_size = generator_output_size
        self.discriminator_output_size = discriminator_output_size
        self.noise_size = noise_size
        self.configs = copy.deepcopy(DEFAULT_GAN_CONFIGS())
        if configs is not None:
            self.configs.update(configs)

        if self.configs["gan_type"] != "lsgan":
            raise RuntimeError("Cuurently only implemented ls_gan.")

        self.generator = Generator(generator_output_size, generator_layers, noise_size, self.configs)
        self.discriminator = Discriminator(generator_output_size, discriminator_layers, discriminator_output_size,
                                           self.configs)

        # This init function mimics the initialization of the TensorFlow implementation that I adapted to Tensorflow
        def weights_init(weight_init_fn, bias_init_fn, m):
            if isinstance(m, torch.nn.Linear):
                weight_init_fn(m.weight)
                bias_init_fn(m.bias)

        self.generator.apply(partial(weights_init, self.configs['generator_weight_initializer'],
                                     torch.nn.init.zeros_))
        self.discriminator.apply(partial(weights_init, self.configs['discriminator_weight_initializer'],
                                         torch.nn.init.zeros_))

        self.generator_optimizer = self.configs["generator_optimizer"](self.generator.parameters(), lr=1e-3)
        self.discriminator_optimizer = self.configs["discriminator_optimizer"](self.discriminator.parameters(), lr=1e-3)

    def sample_random_noise(self, size):
        return np.random.randn(size, self.noise_size)

    def sample_generator(self, size):
        generator_samples = []
        generator_noise = []
        batch_size = self.configs['batch_size']
        for i in range(0, size, batch_size):
            sample_size = min(batch_size, size - i)
            noise = self.sample_random_noise(sample_size)
            generator_noise.append(noise)
            generator_samples.append(
                self.generator.forward(torch.from_numpy(noise).float()).detach().numpy()
            )
        return np.vstack(generator_samples), np.vstack(generator_noise)

    def train(self, X, Y, outer_iters, generator_iters=None, discriminator_iters=None):
        if generator_iters is None:
            generator_iters = self.configs['default_generator_iters']
        if discriminator_iters is None:
            discriminator_iters = self.configs['default_discriminator_iters']

        batch_size = self.configs['batch_size']
        generated_Y = np.zeros((batch_size, self.discriminator_output_size))
        batch_feed_X = batch_feed_array(X, batch_size)
        batch_feed_Y = batch_feed_array(Y, batch_size)

        for i in range(outer_iters):
            for j in range(discriminator_iters):
                sample_X = next(batch_feed_X)
                sample_Y = next(batch_feed_Y)
                generated_X, random_noise = self.sample_generator(batch_size)

                train_X = np.vstack([sample_X, generated_X])
                train_Y = np.vstack([sample_Y, generated_Y])

                dis_log_loss = self.train_discriminator(train_X, train_Y, 1, no_batch=True)

            for i in range(generator_iters):
                gen_log_loss = self.train_generator(random_noise, 1)
                if i > 5:
                    random_noise = self.sample_random_noise(batch_size)

            if i % self.configs['print_iteration'] == 0 and not self.configs['supress_all_logging']:
                print('Iter: {}, generator loss: {}, discriminator loss: {}'.format(i, gen_log_loss, dis_log_loss))

        return dis_log_loss, gen_log_loss

    def train_discriminator(self, X, Y, iters, no_batch=False):
        """
        :param X: goal that we know lables of
        :param Y: labels of those goals
        :param iters: of the discriminator trainig
        The batch size is given by the configs of the class!
        discriminator_batch_noise_stddev > 0: check that std on each component is at least this. (if com: 2)
        """
        if no_batch:
            assert X.shape[0] == Y.shape[0]
            batch_size = X.shape[0]
        else:
            batch_size = self.configs['batch_size']

        batch_feed_X = batch_feed_array(X, batch_size)
        batch_feed_Y = batch_feed_array(Y, batch_size)

        for i in range(iters):
            train_X = torch.from_numpy(next(batch_feed_X)).float()
            train_Y = torch.from_numpy(next(batch_feed_Y)).float()

            preds = self.discriminator.forward(train_X)
            loss = torch.mean(torch.square(2 * train_Y - 1 - preds))

            self.discriminator_optimizer.zero_grad()
            loss.backward()
            self.discriminator_optimizer.step()

        return loss.detach().numpy()

    def train_generator(self, X, iters):
        """
        :param X: These are the latent variables that were used to generate??
        :param iters:
        :return:
        """
        batch_size = self.configs['batch_size']
        batch_feed_X = batch_feed_array(X, batch_size)

        for i in range(iters):
            train_X = torch.from_numpy(next(batch_feed_X)).float()

            generated_samples = self.generator.forward(train_X)
            loss = torch.mean(torch.square(self.discriminator.forward(generated_samples) - 1))

            self.generator_optimizer.zero_grad()
            loss.backward()
            self.generator_optimizer.step()

        return loss.detach().numpy()

    def discriminator_predict(self, X):
        batch_size = self.configs['batch_size']
        output = []
        for i in range(0, X.shape[0], batch_size):
            sample_size = min(batch_size, X.shape[0] - i)
            output.append(self.discriminator.forward(torch.from_numpy(X[i:i + sample_size])).detach().numpy())
        return np.vstack(output)


class Generator(torch.nn.Module):
    def __init__(self, output_size, hidden_layers, noise_size, configs):
        super(Generator, self).__init__()
        self.configs = configs

        layers = []
        input_dim = noise_size
        for size in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, size, bias=True))
            input_dim = size

            if configs['generator_hidden_activation'] == 'relu':
                layers.append(torch.nn.ReLU())
            elif configs['generator_hidden_activation'] == 'leaky_relu':
                layers.append(torch.nn.LeakyReLU(negative_slope=0.2))
            else:
                raise ValueError('Unsupported activation type')

        layers.append(torch.nn.Linear(input_dim, output_size, bias=True))

        if configs['generator_output_activation'] == 'tanh':
            layers.append(torch.nn.Tanh())
        elif configs['generator_output_activation'] == 'sigmoid':
            layers.append(torch.nn.Sigmoid())
        elif configs['generator_output_activation'] != 'linear':
            raise ValueError('Unsupported activation type!')

        self.network = torch.nn.Sequential(*layers)

    def forward(self, noise):
        return self.network.forward(noise)


class Discriminator(torch.nn.Module):

    def __init__(self, input_dim, hidden_layers, output_size, configs):
        super().__init__()
        layers = []
        for i, size in enumerate(hidden_layers):
            layers.append(torch.nn.Linear(input_dim, size, bias=True))
            input_dim = size

            if configs['discriminator_hidden_activation'] == 'relu':
                layers.append(torch.nn.ReLU())
            elif configs['discriminator_hidden_activation'] == 'leaky_relu':
                layers.append(torch.nn.LeakyReLU(0.1))
            else:
                raise ValueError('Unsupported activation type')

        layers.append(torch.nn.Linear(input_dim, output_size, bias=True))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, samples):
        return self.network.forward(samples)
