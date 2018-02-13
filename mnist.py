# Let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
from tensorflow.python import debug as tf_debug


import tensorflow as tf 
import numpy as np
import os
from skimage.io import imsave
from collections import namedtuple

from gan import GAN

fc = tf.contrib.layers.fully_connected
Hook = namedtuple("Hook", ["n", "f"])


class _config(object):
    def __init__(self):
        self.x_batch_size = 256
        self.z_batch_size = 256
        self.z_dims = 100
        self.z_std = 1
        self.num_epochs = 100 
        self.n_critic = 5


def hook_arg_filter(*_args):
    def hook_decorator(f):
        def func_wrapper(*args, **kwargs):
            return f(*[kwargs[arg] for arg in _args])
        return func_wrapper
    return hook_decorator


@hook_arg_filter("g_z", "epoch")
def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    if not os.path.exists("out"):
        os.mkdir("out")
    if not os.path.exists(os.path.join("out", "mnist")):
        os.mkdir(os.path.join("out", "mnist"))

    img_height = img_width = 28
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], \
            img_height, img_width)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255.
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(os.path.join("out", "mnist", "%d.png"%fname), img_grid)


def generator(z):
    with tf.variable_scope("generator"):
        h1 = fc(z, 150, reuse = tf.AUTO_REUSE, scope = "h1")
        h2 = fc(h1, 300, reuse = tf.AUTO_REUSE, scope = "h2")
        h3 = fc(h2, 784, activation_fn = None, \
                reuse = tf.AUTO_REUSE, scope = "h3")
        o = tf.nn.tanh(h3)
            
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


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
real_data = mnist.train.images

# copy by reference?
# TODO: read about Adam
g_optimizer = tf.train.AdamOptimizer(0.0001)
d_optimizer = g_optimizer 

config = _config()
hook1 = Hook(1, show_result)

m = GAN(generator, discriminator, "wasserstein")
# TODO: cleanup code by placing session creation inside .train()
sess = tf.Session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
m.train(sess, g_optimizer, d_optimizer, real_data, config, hooks = [hook1])
