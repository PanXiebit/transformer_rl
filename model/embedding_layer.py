# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

from model import model_utils


class EmbeddingSharedWeights(tf.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, hidden_size, weights_scope_name='embedding_shard_weights'):
        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weights_scope_name = weights_scope_name
        with tf.device("/cpu:0"):
            with tf.variable_scope("softmax" + self.weights_scope_name, reuse=tf.AUTO_REUSE):
                self.embedding_weights = tf.get_variable("softmax_weights", [vocab_size, hidden_size],
                                                     initializer=tf.random_normal_initializer(0., self.hidden_size ** -0.5))

    def call(self, x, need_padding=True):
        """Get token embeddings of x.

        Args:
          x: An int64 tensor with shape [batch_size, length]
        Returns:
          embeddings: float32 tensor with shape [batch_size, length, embedding_size]
          padding: float32 tensor with shape [batch_size, length] indicating the
            locations of the padding tokens in x.
        """
        with tf.name_scope("embedding"):
            embeddings = tf.gather(self.embedding_weights, x)
            # print(embeddings)

            # Scale embedding by the sqrt of the hidden size
            embeddings *= self.hidden_size ** 0.5

            if need_padding:
                # Create binary array of size [batch_size, length]
                # where 1 = padding, 0 = not padding
                padding = model_utils.get_padding(x)

                # Set all padding embedding values to 0
                embeddings *= tf.expand_dims(1 - padding, -1)

            return embeddings

    def linear(self, x):
        """Computes logits by running x through a linear layer.

        Args:
          x: A float32 tensor with shape [batch_size, length, hidden_size]
        Returns:
          float32 tensor with shape [batch_size, length, vocab_size].
        """
        with tf.name_scope("presoftmax_linear"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            x = tf.reshape(x, [-1, self.hidden_size])
            logits = tf.matmul(x, self.embedding_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])


# !!!

class EmbeddingWeights(EmbeddingSharedWeights):
    def __init__(self, vocab_size, hidden_size, weights_scope_name):
        super(EmbeddingWeights, self).__init__(vocab_size, hidden_size, weights_scope_name)

# done
