from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import variables as variables_lib
from tensorflow.contrib.gan.python import losses as tfgan_losses
from tensorflow.contrib.gan.python import namedtuples
from tensorflow.contrib.slim.python.slim import learning as slim_learning
from tensorflow.contrib.training.python.training import training
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.distributions import distribution as ds
from tensorflow.python.ops.losses import losses
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util
from tensorflow.contrib.training.python.training import training
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import supervisor

import os
import sys
import time
import copy

def train_step(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.
  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.
  Returns:
    The total loss and a boolean indicating whether or not to stop training.
  Raises:
    ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  """
  start_time = time.time()

  trace_run_options = None
  run_metadata = None
  summary = None
  if 'should_trace' in train_step_kwargs:
    if 'logdir' not in train_step_kwargs:
      raise ValueError('logdir must be present in train_step_kwargs when '
                       'should_trace is present')
    if sess.run(train_step_kwargs['should_trace']):
      trace_run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
 
  if 'summary_op' not in train_step_kwargs: 
    total_loss, np_global_step = sess.run([train_op, global_step],
                                          options=trace_run_options,
                                          run_metadata=run_metadata)
  else:
    total_loss, np_global_step, summary = sess.run([train_op, global_step, \
                                          train_step_kwargs['summary_op']],
                                          options=trace_run_options,
                                          run_metadata=run_metadata)

  time_elapsed = time.time() - start_time

  if run_metadata is not None:
    tl = timeline.Timeline(run_metadata.step_stats)
    trace = tl.generate_chrome_trace_format()
    trace_filename = os.path.join(train_step_kwargs['logdir'],
                                  'tf_trace-%d.json' % np_global_step)
    logging.info('Writing trace to %s', trace_filename)
    file_io.write_string_to_file(trace_filename, trace)
    if 'summary_writer' in train_step_kwargs:
      train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                           'run_metadata-%d' %
                                                           np_global_step)

  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      logging.info('global step %d: loss = %.4f (%.3f sec/step)',
                   np_global_step, total_loss, time_elapsed)

  # TODO(nsilberman): figure out why we can't put this into sess.run. The
  # issue right now is that the stop check depends on the global step. The
  # increment of global step often happens via the train op, which used
  # created using optimizer.apply_gradients.
  #
  # Since running `train_op` causes the global step to be incremented, one
  # would expected that using a control dependency would allow the
  # should_stop check to be run in the same session.run call:
  #
  #   with ops.control_dependencies([train_op]):
  #     should_stop_op = ...
  #
  # However, this actually seems not to work on certain platforms.
  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False

  return total_loss, should_stop, summary


def get_sequential_train_steps(
    train_steps=namedtuples.GANTrainSteps(1, 1)):
  """Returns a thin wrapper around slim.learning.train_step, for GANs.
  This function is to provide support for the Supervisor. For new code, please
  use `MonitoredSession` and `get_sequential_train_hooks`.
  Args:
    train_steps: A `GANTrainSteps` tuple that determines how many generator
      and discriminator training steps to take.
  Returns:
    A function that can be used for `train_step_fn` for GANs.
  """

  def sequential_train_steps(sess, train_ops, global_step, train_step_kwargs):
    """A thin wrapper around slim.learning.train_step, for GANs.
    Args:
      sess: A Tensorflow session.
      train_ops: A GANTrainOps tuple of train ops to run.
      global_step: The global step.
      train_step_kwargs: Dictionary controlling `train_step` behavior.
    Returns:
      A scalar final loss and a bool whether or not the train loop should stop.
    """
    # Only run `should_stop` at the end, if required. Make a local copy of
    # `train_step_kwargs`, if necessary, so as not to modify the caller's
    # dictionary.
    should_stop_op, train_kwargs = None, train_step_kwargs
    if 'should_stop' in train_step_kwargs:
      should_stop_op = train_step_kwargs['should_stop']
      train_kwargs = train_step_kwargs.copy()
      del train_kwargs['should_stop']

    train_kwargs2 = copy.copy(train_kwargs)
    if 'summary_op' in train_kwargs2:
        del train_kwargs2['summary_op']

    # Run generator training steps.
    gen_loss = 0
    for _ in range(train_steps.generator_train_steps):
      cur_gen_loss, _, _ = train_step(
          sess, train_ops.generator_train_op, global_step, train_kwargs2)
      gen_loss += cur_gen_loss

    # Run discriminator training steps.
    dis_loss = 0
    for _ in range(train_steps.discriminator_train_steps):
      cur_dis_loss, _, summary_d = train_step(
          sess, train_ops.discriminator_train_op, global_step, train_kwargs)
      dis_loss += cur_dis_loss

    sess.run(train_ops.global_step_inc_op)

    # Run the `should_stop` op after the global step has been incremented, so
    # that the `should_stop` aligns with the proper `global_step` count.
    if should_stop_op is not None:
      should_stop = sess.run(should_stop_op)
    else:
      should_stop = False

    return gen_loss + dis_loss, should_stop, summary_d

  return sequential_train_steps
