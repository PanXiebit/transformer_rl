# -*- encoding=utf8 -*-

import tensorflow as tf
from model.base import ModeKeys
from model import attention_layer
from model import embedding_layer
from model import ffn_layer
from model import model_utils
from model import model_params

# params = model_params.TransformerBaseParams()

class EncoderStack(tf.layers.Layer):
    """Transformer encoder stack.

    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
      1. Self-attention layer
      2. Feedforward network (which is 2 fully-connected layers)
    """
    def __init__(self, params, is_train, mode):
        super(EncoderStack, self).__init__()
        self.mode = mode
        self.predict_one = ModeKeys.is_predict_one(self.mode)
        self.layers = []
        for _ in range(params.num_hidden_layers):
            # Create sublayers for each layer.
            self_attention_layer = attention_layer.SelfAttention(
                params.hidden_size, params.num_heads, params.attention_dropout, is_train, self.predict_one)
            feed_forward_network = ffn_layer.FeedFowardNetwork(
                params.hidden_size, params.filter_size, params.relu_dropout, is_train, self.predict_one)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, is_train),
                PrePostProcessingWrapper(feed_forward_network, params, is_train)])
        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(params.hidden_size)

    def call(self, encoder_inputs, attention_bias, inputs_padding):
        """Return the output of the encoder layer stacks.

        Args:
          encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer.
            [batch_size, 1, 1, input_length]
            or None for no need to be added
          inputs_padding: P

        Returns:
          Output of encoder layer stack.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            prepost_self_attention_layer = layer[0]
            prepost_feed_forward_network = layer[1]

            with tf.variable_scope("layer_%d" % n):
                with tf.variable_scope("self_attention"):
                    encoder_inputs = prepost_self_attention_layer(encoder_inputs, attention_bias)
                with tf.variable_scope("ffn"):
                    encoder_inputs = prepost_feed_forward_network(encoder_inputs, inputs_padding)
        return self.output_normalization(encoder_inputs)

class DecoderStack(tf.layers.Layer):
    """Transformer decoder stack.

    Like the encoder stack, the decoder stack is made up of N identical layers.
    Each layer is composed of the sublayers:
      1. Self-attention layer
      2. Multi-headed attention layer combining encoder outputs with results from
         the previous self-attention layer.
      3. Feedforward network (2 fully-connected layers)
    """
    def __init__(self, params, is_train, mode):
        super(DecoderStack, self).__init__()
        self.mode = mode
        self.predict_one = ModeKeys.is_predict_one(self.mode)
        self.layers = []
        for _ in range(params.num_hidden_layers):
            self_attention_layer = attention_layer.SelfAttention(
                params.hidden_size, params.num_heads, params.attention_dropout, is_train, self.predict_one)
            if self.mode == ModeKeys.PREDICT_ONE_DECODER:
                enc_dec_attention_layer = attention_layer.EncDecPredictOneAttention(
                    params.hidden_size, params.num_heads, params.attention_dropout, is_train, self.predict_one)
            else:
                enc_dec_attention_layer = attention_layer.Attention(
                    params.hidden_size, params.num_heads, params.attention_dropout, is_train, self.predict_one)
            feed_forward_network = ffn_layer.FeedFowardNetwork(
                params.hidden_size, params.filter_size, params.relu_dropout, is_train, self.predict_one)
            # decoder 包含3个模块，分别是self-attention,enc_dec_attention,以及feed-forward. 分别wrapper熵layer_norm和dropout.
            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, is_train),
                PrePostProcessingWrapper(enc_dec_attention_layer, params, is_train),
                PrePostProcessingWrapper(feed_forward_network, params, is_train)
            ])
            self.output_normalization = LayerNormalization(params.hidden_size)

    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
             attention_bias, cache=None):
        """Return the output of the decoder layer stacks.
        decoder 部分包含两种attention_bias,encoder-decoder的padding mask和self-attention中的look-ahead mask, padding mask.

        Args:
          decoder_inputs: tensor with shape [batch_size, target_length, hidden_size]
          encoder_outputs: tensor with shape [batch_size, input_length, hidden_size]
          decoder_self_attention_bias: bias for decoder self-attention layer.
            [1, 1, target_len, target_length]
          attention_bias: bias for encoder-decoder attention layer.
            [batch_size, 1, 1, input_length]

          
          cache: (Used for fast decoding) A nested dictionary storing previous
            decoder self-attention values. The items are:
              {layer_n: {"k": tensor with shape [batch_size, i, key_channels],
                         "v": tensor with shape [batch_size, i, value_channels]},
               ...}

        Returns:
          Output of decoder layer stack.
          float32 tensor with shape [batch_size, target_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            prepost_self_attention_layer = layer[0]
            prepost_enc_dec_attention_layer = layer[1]
            prepost_feed_forward_network = layer[2]

            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None

            with tf.variable_scope(layer_name):
                with tf.variable_scope("self-attention"):
                    decoder_inputs = prepost_self_attention_layer(
                        decoder_inputs, decoder_self_attention_bias, cache=layer_cache)
                with tf.variable_scope("encdec_attention"):
                    if self.mode == ModeKeys.PREDICT_ONE_DECODER:
                        decoder_inputs = prepost_enc_dec_attention_layer(
                            decoder_inputs, encoder_outputs, attention_bias, cache=layer_cache)
                    else:
                        decoder_inputs = prepost_enc_dec_attention_layer(
                            decoder_inputs, encoder_outputs, attention_bias)
                with tf.variable_scope("ffn"):
                    decoder_inputs = prepost_feed_forward_network(decoder_inputs)

        return self.output_normalization(decoder_inputs)


class PrePostProcessingWrapper(object):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, params, is_train):
        self.layer = layer
        self.postprocess_dropout = params.layer_postprocess_dropout
        self.train = is_train

        # Create normalization layer
        self.layer_norm = LayerNormalization(params.hidden_size)

    def __call__(self, x, *args, **kwargs):
        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if self.train:
            y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
        return x + y

class LayerNormalization(tf.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                    initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias

if __name__ == "__main__":
    import os
    tf.enable_eager_execution()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    params = model_params.TransformerBaseParams()
    x_inputs = tf.constant([[1,2,3,0,0],[3,4,5,6,8]], dtype=tf.int32)

    Enc_Embedding = embedding_layer.EmbeddingWeights(
                params.source_vocab_size, params.hidden_size, "source_embedding")
    embedded_inputs = Enc_Embedding(x_inputs, not ModeKeys.is_predict_one(ModeKeys.TRAIN))
    print(embedded_inputs.shape)
    attention_bias = model_utils.get_padding_bias(x_inputs)
    print(attention_bias.shape)
    encoder_stack = EncoderStack(params, is_train=True, mode=ModeKeys.TRAIN)
    enc_out = encoder_stack(embedded_inputs, attention_bias, None)
    print(enc_out.shape)
    decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(10)
    self_attention_bias = decoder_self_attention_bias[:, :, 0:1, :1]    
    print(self_attention_bias)
    attention_bias = model_utils.get_padding_bias(x_inputs)
    cache = {
        "layer_%d" % layer: {
            "k": tf.zeros([2, 0, params.hidden_size]),
            "v": tf.zeros([2, 0, params.hidden_size]),
        } for layer in range(params.num_hidden_layers)}
    dec_input = tf.constant([[2], [3]], dtype=tf.int32)
    decoder_stack = DecoderStack(params, is_train=True, mode=ModeKeys.TRAIN)
    dec_out = decoder_stack(dec_input, enc_out, self_attention_bias, attention_bias, cache)
    print(dec_out.shape)

    

    
