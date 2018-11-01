# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simplest FFN model, as described in https://arxiv.org/abs/1611.00421."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .. import model
from .prc import feedback_hgru_3l_temporal

# Note: this model was originally trained with conv3d layers initialized with
# TruncatedNormalInitializedVariable with stddev = 0.01.
def _predict_object_mask(net, depth=9):
  """Computes single-object mask prediction."""
  net = tf.contrib.layers.conv3d(net,
                                 scope='conv0_a',
                                 num_outputs=16,
                                 kernel_size=(3, 3, 3),
                                 padding='SAME')

  from .prc import feedback_hgru_drew
  from .prc import gradients
  with tf.variable_scope('recurrent'):
      net = tf.nn.elu(net)
      hgru = feedback_hgru_drew.hGRU(
          layer_name='hgru_net',
          x_shape=net.get_shape().as_list(),
          timesteps=8,
          h_ext=[[1, 7, 7], [3, 7, 7], [3, 7, 7], [1, 1, 1], [1, 1, 1]],
          strides=[1, 1, 1, 1, 1],
          pool_strides=[1, 4, 4],
          padding='SAME',
          aux={
              'symmetric_weights': True, ## Setting it to True return error
              'dilations': [
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1]
              ],
              'batch_norm': True,
              'dtype': tf.float32,  # tf.bfloat16,
              'pooling_kernel': [1, 4, 4],
              'intermediate_ff': [16,16],  # + filters,
              'intermediate_ks': [[1,5,5], [1,5,5]]},
          train=True)
      net = hgru.build(net)
  net = tf.contrib.layers.batch_norm(
      inputs=net,
      scale=True,
      center=True,
      fused=True,
      renorm=False,
      param_initializers={
            'moving_mean': tf.constant_initializer(0., dtype=tf.float32),
            'moving_variance': tf.constant_initializer(1., dtype=tf.float32),
            'gamma': tf.constant_initializer(0.1, dtype=tf.float32)
            },
      updates_collections=None,
      is_training=True)
  net = tf.contrib.layers.conv3d(net,
                                    scope='conv_lom',
                                    num_outputs=1,
                                    kernel_size=(1, 1, 1),
                                    activation_fn=None)
  return net


class ConvStack3DFFNModel(model.FFNModel):
  dim = 3

  def __init__(self, fov_size=None, deltas=None, batch_size=None, depth=9):
    super(ConvStack3DFFNModel, self).__init__(deltas, batch_size)
    self.set_uniform_io_size(fov_size)
    self.depth = depth

  def define_tf_graph(self):
    self.show_center_slice(self.input_seed)

    if self.input_patches is None:
      self.input_patches = tf.placeholder(
          tf.float32, [1] + list(self.input_image_size[::-1]) +[1],
          name='patches')

    net = tf.concat([self.input_patches, self.input_seed], 4)

    with tf.variable_scope('seed_update', reuse=False):
      logit_update = _predict_object_mask(net, self.depth)

    logit_seed = self.update_seed(self.input_seed, logit_update)

    # Make predictions available, both as probabilities and logits.
    self.logits = logit_seed
    self.logistic = tf.sigmoid(logit_seed)

    if self.labels is not None:
      self.set_up_sigmoid_pixelwise_loss(logit_seed)
      self.set_up_optimizer()
      self.show_center_slice(logit_seed)
      self.show_center_slice(self.labels, sigmoid=False)
      self.add_summaries()

    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
