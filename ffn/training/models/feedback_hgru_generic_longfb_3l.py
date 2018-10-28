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

# Note: this model was originally trained with conv3d layers initialized with
# TruncatedNormalInitializedVariable with stddev = 0.01.
def _predict_object_mask(input_patches, input_seed, depth=9):
  """Computes single-object mask prediction."""
  x = tf.contrib.layers.conv3d(tf.concat([input_patches], axis=4),
                                 scope='conv0_a',
                                 num_outputs=8,
                                 kernel_size=(3, 3, 3),
                                 padding='SAME')

  from .prc import feedback_hgru_generic_longfb_3l
  with tf.variable_scope('recurrent'):
      hgru_net = feedback_hgru_generic_longfb_3l.hGRU(layer_name='hgru_net',
                                                        num_in_feats=8,
                                                        timesteps=3,
                                                        h_repeat=2,
                                                        hgru_dhw=[[3, 7, 7], [3, 5, 5], [1, 3, 3], [1, 1, 1]],
                                                        hgru_k=[8, 16, 32, 8],
                                                        ff_conv_dhw=[[1, 1, 1], [2, 5, 5], [2, 3, 3], [1, 3, 3]],
                                                        ff_conv_k=[8, 16, 32, 48],
                                                        ff_conv_strides=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1],[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                                                        ff_pool_dhw=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                                                        ff_pool_strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                                                        fb_mode='transpose',
                                                        fb_dhw=[[3, 6, 6], [3, 4, 4], [1, 4, 4]],
                                                        fb_k=[16, 32, 48],
                                                        padding='SAME',
                                                        batch_norm=True,
                                                        aux=None,
                                                        train=True)

      net = hgru_net.build(x, input_seed)

  logits = tf.contrib.layers.conv3d(net,
                                    scope='conv_lom',
                                    num_outputs=1,
                                    kernel_size=(1, 1, 1),
                                    activation_fn=None)
  import numpy as np
  acc = 0
  for x in tf.trainable_variables():
      prod = np.prod(x.get_shape().as_list())
      acc += prod
  print('>>>>>>>>>>>>>>>>>>>>>>TRAINABLE VARS: '+str(acc))
  return logits


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

    with tf.variable_scope('seed_update', reuse=False):
      logit_update = _predict_object_mask(self.input_patches, self.input_seed, self.depth)

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
