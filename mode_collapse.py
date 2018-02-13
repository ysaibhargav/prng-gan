import tensorflow as tf 
import numpy as np
import os
from skimage.io import imsave
from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns

from gan import GAN

fc = tf.contrib.layers.fully_connected
Hook = namedtuple("Hook", ["n", "f"])


class _config(object):
    def __init__(self):
        self.x_batch_size = 256
        self.z_batch_size = 256
        self.z_dims = 100
        self.z_std = 1
        self.num_epochs = 1000 


def hook_arg_filter(*_args):
    def hook_decorator(f):
        def func_wrapper(*args, **kwargs):
            return f(*[kwargs[arg] for arg in _args])
        return func_wrapper
    return hook_decorator


@hook_arg_filter("g_z", "d_g_z", "epoch")
def plot_density(g_z, d_g_z, epoch):
    if not os.path.exists("out"):
        os.mkdir("out")
    if not os.path.exists(os.path.join("out", "mc")):
        os.mkdir(os.path.join("out", "mc"))
    g_z = g_z.flatten()
    d_g_z = d_g_z.flatten()

    sorted_pairs = sorted(zip(g_z, d_g_z), key = lambda x: x[0])
    sorted_pairs = np.array(sorted_pairs)
    g_z = sorted_pairs[:, 0]
    d_g_z = sorted_pairs[:, 1]
    sns.set_style('whitegrid')
    plt.plot(g_z, d_g_z)
    sns.distplot(g_z)
    sns.distplot(real_data_for_plot, bins = 20)
    plt.savefig(os.path.join("out", "mc", '%d.png'%epoch))
    plt.close()


def generator(z):
    with tf.variable_scope("generator"):
        # fc parameter type dependent on input data's?
        h1 = fc(z, 150, reuse = tf.AUTO_REUSE, scope = "h1")
        h2 = fc(h1, 300, reuse = tf.AUTO_REUSE, scope = "h2")
        h3 = fc(h2, 1, activation_fn = None, \
                reuse = tf.AUTO_REUSE, scope = "h3")
        #o = tf.nn.tanh(h3)
        o = h3
            
        return o

# TODO: dropout
def discriminator(x):
    with tf.variable_scope("discriminator"):
        h1 = fc(x, 200, reuse = tf.AUTO_REUSE, scope = "h1")
        h2 = fc(h1, 150, reuse = tf.AUTO_REUSE, scope = "h2")
        h3 = fc(h2, 1, activation_fn = None, \
                reuse = tf.AUTO_REUSE, scope = "h3")
        o = tf.nn.sigmoid(h3)

        return o

# mixture of gaussians dataset
mode1 = np.random.randn(2000) * 1
mode2 = np.random.randn(2000) * 1 + 10
mode3 = np.random.randn(2000) * 1 + 20
mode4 = np.random.randn(2000) * 1 + 30
mode5 = np.random.randn(2000) * 1 + 40
#real_data = np.hstack([mode1, mode2, mode3, mode4, mode5])
real_data = np.hstack([mode1, mode3, mode5])
real_data = np.array(real_data, dtype = np.float32)
real_data_for_plot = np.random.choice(real_data, 250)
real_data = real_data[:, None]

# copy by reference?
# TODO: read about Adam
g_optimizer = tf.train.AdamOptimizer(0.0001)
d_optimizer = g_optimizer 

config = _config()
hook1 = Hook(1, plot_density)

m = GAN(generator, discriminator, "vanilla")
# TODO: cleanup code by placing session creation inside .train()
sess = tf.Session()
m.train(sess, g_optimizer, d_optimizer, real_data, config, hooks = [hook1])
