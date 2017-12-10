from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import glob
import pdb
import lib

from tensorflow.contrib.training.python.training import training
from tensorflow.contrib.gan.python import namedtuples

tf.logging.set_verbosity(tf.logging.INFO)

layers = tf.contrib.layers
tfgan = tf.contrib.gan
slim = tf.contrib.slim

batch_size = 64
noise_dims = 200
learning_rate = 1e-3
num_epochs = 5
max_number_of_steps = None 
num_batches = None
output_len = 100
num_g_steps, num_d_steps = 1, 1
img_h = img_w = int(np.sqrt(output_len))
max_global_norm = 5
train_log_dir = "rnn-log"
data_path = os.path.join("..", "data", "true")
logs_path = os.path.join("logs")

g_conf = {"num_layers": 1, "num_units": 200}
d_conf = {"num_layers": 1, "num_units": 100, "num_hidden": 128}

def data_loader(data_path):
    global max_number_of_steps
    global num_batches
    data = []
    for f in glob.glob(os.path.join(data_path, "*.txt")):
        _data = list(np.loadtxt(f))
        data = data + _data

    data = data[:int(len(data) / output_len) * output_len]
    data = np.array(data)
    data = np.resize(data, (-1, output_len))
    num_batches = int(len(data) / batch_size)
    data = data[:num_batches * batch_size]
    if max_number_of_steps is None:
        max_number_of_steps = num_epochs * num_batches
        max_number_of_steps *= num_d_steps
        print(max_number_of_steps)

    data = tf.convert_to_tensor(data, dtype = tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.repeat(num_epochs * num_d_steps)
    dataset = dataset.batch(batch_size)
    
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    return next_element

# TODO: implement seq2seq based generator
def generator(noise):
    with tf.name_scope("generator"):
        def lstm_cell(cell_type):
            with tf.name_scope(cell_type):
                return tf.contrib.rnn.BasicLSTMCell(num_units = \
                    g_conf["num_units"])
        
        cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell("fw") \
            for _ in range(g_conf["num_layers"])])
        cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell("bw") \
            for _ in range(g_conf["num_layers"])])

        noise = tf.unstack(noise, axis = 1)
        outputs, _, _ = tf.nn.static_bidirectional_rnn(
            cell_fw, cell_bw, noise, dtype = tf.float32)

        outputs = tf.stack(outputs, axis = 1)
        flattened_outputs = layers.flatten(outputs)

        output = layers.fully_connected(
            flattened_outputs, output_len, activation_fn = tf.nn.sigmoid)

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
        def lstm_cell(cell_type):
            with tf.name_scope(cell_type):
                return tf.contrib.rnn.BasicLSTMCell(num_units = \
                    d_conf["num_units"])

        cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell("fw") \
            for _ in range(d_conf["num_layers"])])
        cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell("bw") \
            for _ in range(d_conf["num_layers"])])

        _input = tf.expand_dims(_input, -1)
        _input = tf.unstack(_input, axis = 1)

        outputs, _, _ = tf.nn.static_bidirectional_rnn(
            cell_fw, cell_bw, _input, dtype = tf.float32)

        outputs = tf.stack(outputs, axis = 1)
        flattened_outputs = layers.flatten(outputs)

        logits = layers.fully_connected(
            flattened_outputs, d_conf["num_hidden"], \
            activation_fn = tf.nn.relu)

        logits_real = layers.linear(logits, 1)

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
        [batch_size, noise_dims, 1]))

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
        generator_optimizer=tf.train.AdamOptimizer(gen_lr),
        discriminator_optimizer=tf.train.AdamOptimizer(dis_lr),
        summarize_gradients=True,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
        transform_grads_fn=training.clip_gradient_norms_fn(max_global_norm))

train_step_fn = lib.get_sequential_train_steps(\
    namedtuples.GANTrainSteps(num_g_steps, num_d_steps))

global_step = tf.train.get_or_create_global_step()
loss_values= []

for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    summary_writer = tf.summary.FileWriter(logs_path, \
        graph = tf.get_default_graph())

    sess.run(tf.global_variables_initializer())
    with slim.queues.QueueRunners(sess):
        for i in xrange(max_number_of_steps):
            print("Global step %d"%i)
            cur_loss, _, summary = train_step_fn(
                sess, train_ops, global_step, train_step_kwargs={\
                    'summary_op': merged_summary_op})

            summary_writer.add_summary(summary, i)
            loss_values.append((i, cur_loss))

