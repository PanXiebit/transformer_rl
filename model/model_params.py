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
"""Defines Transformer model parameters."""

class TransformerBaseParams(object):
    """Parameters for the base Transformer model."""
    # Input params
    batch_size = 4096 # Maximum number of tokens per batch of examples.
    max_length = 256  # Maximum number of tokens per example.

    # Model params
    initializer_gain = 1.0  # Used in trainable variable initialization.
    vocab_size = 30000  # Number of tokens defined in the vocabulary file.
    hidden_size = 256  # Model dimension in the hidden layers.
    num_hidden_layers = 4  # Number of layers in the encoder and decoder stacks.
    num_heads = 4  # Number of heads to use in multi-headed attention.
    filter_size = 1536  # 2048  # Inner layer dimensionality in the feedforward network.
    # !!!
    source_vocab_size = vocab_size
    target_vocab_size = vocab_size

    shared_embedding_softmax_weights = False
    # done

    # Dropout values (only used when training)
    layer_postprocess_dropout = 0.1
    attention_dropout = 0.1
    relu_dropout = 0.1

    # Training params
    label_smoothing = 0.1
    learning_rate = 2.0
    learning_rate_decay_rate = 1.0
    learning_rate_warmup_steps = 8000

    # Optimizer params
    optimizer_adam_beta1 = 0.9
    optimizer_adam_beta2 = 0.997
    optimizer_adam_epsilon = 1e-09

    # Default prediction params
    extra_decode_length = 20
    beam_size = 1
    alpha = 0.6  # used to calculate length normalization in beam search


class TransformerSmallParams(TransformerBaseParams):
    """Parameters for the big Transformer model."""
    batch_size = 4096
    hidden_size = 128
    filter_size = 768
    num_heads = 1
    num_hidden_layers = 2


class TransformerBigParams(TransformerBaseParams):
    """Parameters for the big Transformer model."""
    batch_size = 4096
    hidden_size = 1024
    filter_size = 4096
    num_heads = 16
    num_hidden_layers = 6


class TransformerBeamMid1Shard4Params(TransformerBaseParams):
    """Parameters for the big Transformer model."""
    num_hidden_layers = 4  # Number of layers in the encoder and decoder stacks.
    batch_size = 4096
    hidden_size = 256
    filter_size = 1536
    num_heads = 4


class TransformerBeamMid1Shard4Dropout0Params(TransformerBeamMid1Shard4Params):
    """Parameters for the big Transformer model."""
    layer_postprocess_dropout = 0
    attention_dropout = 0
    relu_dropout = 0


class TransformerBeamMid1ShardTest4(TransformerBaseParams):
    """Parameters for the big Transformer model."""
    layer_postprocess_dropout = 0
    attention_dropout = 0
    relu_dropout = 0
    num_hidden_layers = 4
    batch_size = 4096
    hidden_size = 256
    filter_size = 2048
    num_heads = 8

class TransformerBeamMid1ShardTest6(TransformerBaseParams):
    """Parameters for the big Transformer model."""
    layer_postprocess_dropout = 0
    attention_dropout = 0
    relu_dropout = 0
    num_hidden_layers = 4
    batch_size = 4096
    hidden_size = 512
    filter_size = 2048
    num_heads = 4
    embedding_size = 512



