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
"""Implementation of multiheaded attention and self-attention layers."""

import tensorflow as tf


class Attention(tf.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, attention_dropout, is_train, predict_one=False):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.is_train = is_train
        self.predict_one = predict_one

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
        self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
        self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                                  name="output_transform")

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.

        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.

        Args:
          x: A tensor with shape [batch_size, length, hidden_size]

        Returns:
          A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension
            if self.predict_one:
                x = tf.reshape(x, [1, -1, self.num_heads, depth])
            else:
                x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.

        Args:
          x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

        Returns:
          A tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            if self.predict_one:
                x = tf.reshape(x, [1, -1, self.hidden_size])
            else:
                x = tf.reshape(x, [batch_size, length, self.hidden_size])
            return x

    def call(self, x, y, bias, cache=None):
        """Apply attention mechanism to x and y.

        Args:
          x: a tensor with shape [batch_size, length_x, hidden_size]
          y: a tensor with shape [batch_size, length_y, hidden_size]
          bias: attention bias that will be added to the result of the dot product.
            if bias is None, means no need to be added, usually means in PREDICT_ONE mode.
          if padding mask:
            bias.shape = [batch, 1, 1, length].
          if self-attention look-ahead mask:
            bias.shape = [1, 1, q_length, kv_length]


          cache: (Used during prediction) dictionary with tensors containing results
            of previous attentions. The dictionary must have the items:
                {"k": tensor with shape [batch_size, i, key_channels],
                 "v": tensor with shape [batch_size, i, value_channels]}
            where i is the current decoded length.

        Returns:
          Attention layer output with shape [batch_size, length_x, hidden_size]
        """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).

        if self.predict_one:
            x = tf.reshape(x, [-1, self.hidden_size])
            y = tf.reshape(y, [-1, self.hidden_size])

        q = self.q_dense_layer(x)
        #print("query.shape:{}".format(q.shape))
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if self.predict_one:
            q = tf.reshape(q, [1, -1, self.hidden_size])
            k = tf.reshape(k, [1, -1, self.hidden_size])
            v = tf.reshape(v, [1, -1, self.hidden_size])

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)
            # print("cache, k, v", k.shape, v.shape)
            # Update cache
            cache["k"] = k
            cache["v"] = v
            #tf.logging.info("cache {}".format(cache["k"].shape))
        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        #tf.logging.info("x.shape {}, y.shape {}".format(x.shape, y.shape))
        return self.compute_attention(q, k, v, bias)

    def compute_attention(self, q, k, v, bias):
        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        # Calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)   # [batch, heads, q_len, kv_len]
        # tf.logging.info("attention_logits:{}".format(logits.shape))
        #tf.logging.info("logits: {}, bias: {}".format(logits.shape, bias.shape))
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights") # [batch, heads, q_len, kv_len]
        if self.is_train:
            weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
        
        attention_output = tf.matmul(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        if self.predict_one:
            attention_output = tf.reshape(attention_output, [-1, self.hidden_size])
        attention_output = self.output_dense_layer(attention_output)
        if self.predict_one:
            attention_output = tf.reshape(attention_output, [1, -1, self.hidden_size])
        return attention_output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, x, bias, cache=None):
        return super(SelfAttention, self).call(x, x, bias, cache)


class EncDecPredictOneAttention(Attention):
    """
    Only used in predict_one_decoder graph building
    """

    def __init__(self, hidden_size, num_heads, attention_dropout, is_train, predict_one=False):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")

        super(Attention, self).__init__(name='attention')
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.is_train = is_train
        self.predict_one = predict_one

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")

        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                                  name="output_transform")

    def call(self, x, y, bias, cache):
        """Must match attention call !!! """
        x = tf.reshape(x, [-1, self.hidden_size]) # []
        q = self.q_dense_layer(x)
        q = tf.reshape(q, [1, -1, self.hidden_size])
        q = self.split_heads(q)

        k = cache['encdec_k']
        v = cache['encdec_v']
        print("cache, key:{}, value:{}".format(k.shape, v.shape))
        return self.compute_attention(q, k, v, bias)


if __name__ == "__main__":
    multi_attn = Attention(hidden_size=512,
                           num_heads=8,
                           attention_dropout=0.1,
                           is_train=False,
                           predict_one=False)

