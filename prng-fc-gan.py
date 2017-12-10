"""
FULLY-CONNECTED GAN
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.gan.python import namedtuples

import tensorflow as tf
import numpy as np
import os
import glob
import pdb
import lib

tf.logging.set_verbosity(tf.logging.INFO)

layers = tf.contrib.layers
tfgan = tf.contrib.gan
slim = tf.contrib.slim

batch_size = 128
noise_dims = 10000
learning_rate = 1e-3
num_epochs = 20
max_number_of_steps = None 
num_batches = None
output_len = 4096
num_g_steps, num_d_steps = 1, 1
img_h = img_w = int(np.sqrt(output_len))
train_log_dir = "log"
data_path = os.path.join("..", "data", "true")
logs_path = os.path.join("logs")
real_data = None

def data_loader(data_path):
    global max_number_of_steps
    global num_batches
    global real_data
    data = []
    for f in glob.glob(os.path.join(data_path, "*.txt")):
        _data = list(np.loadtxt(f))
        data = data + _data

    data = data[:int(len(data) / output_len) * output_len]
    data = np.array(data)
    data = np.resize(data, (-1, output_len))
    real_data = data[:batch_size]
    num_batches = int(len(data) / batch_size)
    data = data[:num_batches * batch_size]
    if max_number_of_steps is None:
        max_number_of_steps = num_epochs * num_batches
        max_number_of_steps *= num_d_steps

    data = tf.convert_to_tensor(data, dtype = tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.repeat(num_epochs * num_d_steps)
    dataset = dataset.batch(batch_size)
    
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    # dims [batch_size, output_len]
    return next_element


def generator(noise):
    with tf.name_scope("generator"):
        num_units_array = []

        _input = noise
        i = -1
        for i, num_units in enumerate(num_units_array):
            tf.summary.histogram("g_h_pre%i"%i, _input)
            _input = layers.fully_connected(_input, num_units, 
                activation_fn = tf.nn.relu)
            tf.summary.histogram("g_h_act%i"%i, _input)

        i += 1
        tf.summary.histogram("g_h_pre%i"%i, _input)
        output = layers.fully_connected(_input, output_len, 
            activation_fn = tf.nn.sigmoid)
        tf.summary.histogram("g_h_act%i"%i, _input)

        output_image = tf.reshape(output[0], [-1])
        thr = tf.constant(0.5)
        def one(): return 1.
        def zero(): return 0.
        output_image = tf.map_fn(lambda y: tf.cond(tf.less_equal(thr, y), \
            one, zero), output_image)
        output_image = tf.reshape(output_image, [-1, img_w, img_h, 1])
        tf.summary.image("g_z", output_image)#, 3*max_number_of_steps)

        return output

def discriminator(_input, _):
    with tf.name_scope("discriminator"):
        num_units_array = [500, 500]

        for i, num_units in enumerate(num_units_array):
            tf.summary.histogram("d_h_pre%i"%i, _input)

            _input = layers.fully_connected(_input, num_units, \
                activation_fn = tf.nn.relu)

            tf.summary.histogram("d_h_act%i"%i, _input)

        logits = _input

        logits_real = layers.linear(logits, 1)
        tf.summary.histogram("d_out", logits_real)
        tf.summary.scalar("d_stats", tf.reduce_mean(logits_real))

        return logits_real      

if tf.gfile.Exists(logs_path):
    tf.gfile.DeleteRecursively(logs_path)
else:
    tf.gfile.MkDir(logs_path)

with tf.device('/cpu:0'):
    data = data_loader(data_path)

gan_model = tfgan.gan_model(
    generator_fn = generator,
    discriminator_fn = discriminator, 
    real_data = data,
    generator_inputs = tf.random_normal(
        [batch_size, noise_dims]))

with tf.name_scope("loss"):
    gan_loss = tfgan.gan_loss(
        gan_model,
        gradient_penalty_weight = 1,
        add_summaries = True)

with tf.name_scope("train"):
    gen_lr, dis_lr = learning_rate, learning_rate
    train_ops = tfgan.gan_train_ops(
        gan_model,
        gan_loss,
        generator_optimizer=tf.train.GradientDescentOptimizer(gen_lr),
        discriminator_optimizer=tf.train.GradientDescentOptimizer(dis_lr),
        summarize_gradients=True,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    status_message = tf.string_join(
        ['Starting train step: ',
        tf.as_string(tf.train.get_or_create_global_step())],
        name='status_message')

    """tfgan.gan_train(
        train_ops,
        hooks=[tf.train.StopAtStepHook(num_steps = max_number_of_steps),
            tf.train.LoggingTensorHook([status_message], every_n_iter=10)],
        logdir=train_log_dir,
        get_hooks_fn=tfgan.get_joint_train_hooks())"""

#train_step_fn = tfgan.get_sequential_train_steps()
train_step_fn = lib.get_sequential_train_steps(\
    namedtuples.GANTrainSteps(num_g_steps, num_d_steps))
global_step = tf.train.get_or_create_global_step()

for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

merged_summary_op = tf.summary.merge_all()

loss_values = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(logs_path, \
        graph = tf.get_default_graph())

    with slim.queues.QueueRunners(sess):
        for i in xrange(max_number_of_steps):
            print("Global step %d"%i)
            cur_loss, _, summary = train_step_fn(
                sess, train_ops, global_step, train_step_kwargs={\
                    'summary_op': merged_summary_op})
            """
            cur_loss, _ = train_step_fn(
                sess, train_ops, global_step, train_step_kwargs={})
            """
            summary_writer.add_summary(summary, i)

            loss_values.append((i, cur_loss))
            if i == max_number_of_steps - 1: 
                with tf.variable_scope('Generator', reuse=True):
                    g_z = gan_model.generator_fn(tf.random_normal(\
                        [batch_size, noise_dims]))    

                """
                with tf.variable_scope('Discriminator', reuse=True):
                    d_g_z = gan_model.discriminator_fn(g_z, 0)

                with tf.variable_scope('Discriminator', reuse=True):
                    d_input = tf.placeholder(tf.float32, shape=(None, output_len))
                    d_x = gan_model.discriminator_fn(d_input, 0)
                """
                    
                """g_z, d_g_z, d_x = sess.run([g_z, d_g_z, d_x], \
                    feed_dict = {d_input: real_data})"""

                g_z = sess.run(g_z)
    
                # bitmap
                g_z[g_z >= 0.5] = 1 
                g_z[g_z < 0.5] = 0 
                """
                # discriminator stats
                d_stats_z = (sum(d_g_z <= 0) + 0.) / len(d_g_z)
                d_stats_x = (sum(d_x > 0) + 0.) / len(d_x)
                
                print("D(G(z)): %f"%d_stats_z)
                print("D(x): %f"%d_stats_x)
                """

                for j, val in enumerate(g_z):
                    """
                    with open(os.path.join("out", \
                        "%d-%d.txt"%(i/num_batches, j)), 'w') as f:
                        f.writelines("\n".join(["%d"%_ for _ in val]))    

                    with open(os.path.join("out", \
                        "stats-%d-%d.txt"%(i/num_batches, j)), 'w') as f:
                        f.writelines("%.4f %.4f %.4f\n"%(float(sum(val)) \
                            / output_len, d_stats_z, d_stats_x))    
                    """
                    with open(os.path.join("fcgan-imgs", "%d.txt"%j), \
                        'w') as f:
                        f.writelines("\n".join(["%d"%_ for _ in val]))

