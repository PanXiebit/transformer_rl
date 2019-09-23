from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import nest
from model.base import ModeKeys
from model import embedding_layer
from model import model_utils
from model import model_params
from model import EncoderDecoder
from utils import metrics
from utils.tokenizer import EOS_ID, PAD_ID
import time


class Transformer(object):
    def __init__(self, params, is_train, mode=None):
        self.is_train = is_train
        self.params = params

        if mode is not None:
            self.mode = mode
        elif self.is_train:
            self.mode = ModeKeys.TRAIN
        else:
            self.mode = ModeKeys.PREDICT

        self.encoder_stack = EncoderDecoder.EncoderStack(params, is_train, self.mode)
        self.decoder_stack = EncoderDecoder.DecoderStack(params, is_train, self.mode)
        self._initializer = tf.variance_scaling_initializer(
            self.params.initializer_gain, mode="fan_avg", distribution="uniform")

    def init_embed(self, name_scope):
        with tf.variable_scope(name_scope, initializer=self._initializer, reuse=tf.AUTO_REUSE):
            if self.params.shared_embedding_softmax_weights:
                print("sharing embedding!!!")
                self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
                    self.params.vocab_size, self.params.hidden_size)
                self.encoder_embedding_layer = self.embedding_softmax_layer
                self.decoder_embedding_layer = self.embedding_softmax_layer
                self.decoder_softmax_layer = self.embedding_softmax_layer
            else:
                print("not sharing embedding!!!")
                self.encoder_embedding_layer = embedding_layer.EmbeddingWeights(
                    self.params.source_vocab_size, self.params.hidden_size, "source_embedding")
                self.decoder_embedding_layer = embedding_layer.EmbeddingWeights(
                    self.params.target_vocab_size, self.params.hidden_size, "target_embedding")
                self.decoder_softmax_layer = embedding_layer.EmbeddingWeights(
                    self.params.target_vocab_size, self.params.hidden_size, 'soft_max')


    def build_pretrain(self, inputs, targets):
        self.init_embed("Transformer")
        with tf.variable_scope("Transformer", initializer=self._initializer, reuse=tf.AUTO_REUSE):
            attention_bias = model_utils.get_padding_bias(inputs)  # [batch, 1, 1, src_len]
            encoder_outputs = self.encode(inputs, attention_bias)  # [batch, src_len, hidden_size]
            if targets is None:
                prediction = self.argmax_predict(encoder_outputs, attention_bias)
                return prediction
            else:
                tf.logging.info("!!!!!!!!!! pretrain decoder !!!!!!!!!!!!!!!!!!")
                logits = self.decode(targets, encoder_outputs, attention_bias)  # [batch, tgt_len, vocab_size]
                return logits

    #def get_real_loss(self, origin_inputs, origin_target):
    #    with tf.variable_scope("Transformer", initializer=self._initializer, reuse=tf.AUTO_REUSE):
    #        real_attention_bias = model_utils.get_padding_bias(origin_target)  # [batch, 1, 1, src_len]
    #        real_encoder_outputs = self.encode(origin_target, real_attention_bias)  # [batch, src_len, hidden_size]
    #        real_logits = self.decode(origin_inputs, real_encoder_outputs, real_attention_bias)
    #        real_xentropy, real_weights = metrics.padded_cross_entropy_loss(
    #            real_logits, origin_inputs, self.params.label_smoothing, self.params.target_vocab_size)
    #        self.real_loss = tf.reduce_sum(real_xentropy) / tf.reduce_sum(real_weights)  # [batch]
    #        return self.real_loss

    def encode(self, inputs, attention_bias):
        """Generate continuous representation for inputs."""
        with tf.name_scope("encode"):
            embedded_inputs = self.encoder_embedding_layer(inputs, not ModeKeys.is_predict_one(self.mode))
            inputs_padding = model_utils.get_padding(inputs)
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(embedded_inputs)[1]
                if ModeKeys.is_predict_one(self.mode):
                    pos_encoding = model_utils.get_position_encoding(
                        self.params.max_length, self.params.hidden_size
                    )
                    pos_encoding = tf.slice(pos_encoding, [0, 0], [length, self.params.hidden_size],
                                            name='slice_pos_encoding')
                else:
                    pos_encoding = model_utils.get_position_encoding(
                        length, self.params.hidden_size)

                encoder_inputs = embedded_inputs + pos_encoding

            if self.is_train:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, 1 - self.params.layer_postprocess_dropout)

            return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)

    def decode(self, targets, encoder_outputs, attention_bias):
        """Generate logits for each value in the target sequence."""
        with tf.name_scope("decode"):
            decoder_inputs = self.decoder_embedding_layer(targets, not ModeKeys.is_predict_one(self.mode))
            with tf.name_scope("shift_targets"):
                decoder_inputs = tf.pad(
                    decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # [batch, tgt_seqn_len, embed_size]
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                decoder_inputs += model_utils.get_position_encoding(
                    length, self.params.hidden_size)
            if self.is_train:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, 1 - self.params.layer_postprocess_dropout)

            # Run values
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
                length)
            outputs = self.decoder_stack(
                decoder_inputs, encoder_outputs, decoder_self_attention_bias,
                attention_bias)
            logits = self.decoder_softmax_layer.linear(outputs)
            return logits

    def argmax_predict(self, encoder_outputs, encoder_decoder_attention_bias):
        if ModeKeys.is_predict_one(self.mode):
            batch_size = 1
        else:
            batch_size = tf.shape(encoder_outputs)[0]
        input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = input_length + self.params.extra_decode_length

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

        cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.params.hidden_size]),
                "v": tf.zeros([batch_size, 0, self.params.hidden_size]),
            } for layer in range(self.params.num_hidden_layers)}

        cache["encoder_outputs"] = encoder_outputs
        if not ModeKeys.is_predict_one(self.mode):
            cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

        if self.params.beam_size > 1:
            pass
        else:
            def inner_loop(i, finished, next_id, decoded_ids, cache):
                logits, cache = symbols_to_logits_fn(next_id, i, cache)  # [batch, vocab_size]
                next_id = tf.argmax(logits, -1, output_type=tf.int32)
                finished |= tf.equal(next_id, EOS_ID)
                next_id = tf.reshape(next_id, shape=[-1, 1])
                decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
                return i + 1, finished, next_id, decoded_ids, cache

            def is_not_finished(i, finished, _1, _2, _3):
                return (i < max_decode_length) & tf.logical_not(tf.reduce_all(finished))

            decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int32)
            finished = tf.fill([batch_size], False)
            next_id = tf.zeros([batch_size, 1], dtype=tf.int32)

            _, _, _, decoded_ids, _ = tf.while_loop(
                cond=is_not_finished,
                body=inner_loop,
                loop_vars=[tf.constant(0), finished, next_id, decoded_ids, cache],
                shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    nest.map_structure(get_state_shape_invariants, cache),
                ])
            return decoded_ids

    def _get_symbols_to_logits_fn(self, max_decode_length):
        if ModeKeys.is_predict_one(self.mode):
            timing_signal = model_utils.get_position_encoding(
                self.params.max_length, self.params.hidden_size
            )
            timing_signal = tf.slice(timing_signal, [0, 0], [max_decode_length + 1, self.params.hidden_size],
                                     name='slice_timing_signal')
        else:
            timing_signal = model_utils.get_position_encoding(
                max_decode_length + 1, self.params.hidden_size)  # [max_decode_length + 1, hidden_size]

        if ModeKeys.is_predict_one(self.mode):
            decoder_self_attention_bias = None
        else:
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
                max_decode_length)  # [1, 1, max_decode_length, max_decode_length]

        def symbols_to_logits_fn(ids, i, cache):

            decoder_input = ids[:, -1:]  # [batch, 1]

            decoder_input = self.decoder_embedding_layer(decoder_input, not ModeKeys.is_predict_one(
                self.mode))  # [batch, 1, hidden_size]
            if ModeKeys.is_predict_one(self.mode):
                decoder_input = decoder_input * (1 - tf.to_float(tf.equal(i, 0)))

            slice_pos_encoding = tf.slice(timing_signal, [i, 0], [1, self.params.hidden_size],
                                          name='slice_pos_encoding')  # [1, hidden_size]
            decoder_input += slice_pos_encoding

            if decoder_self_attention_bias is None:
                self_attention_bias = None
            else:
                self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]  # [1, 1, 1, time_step]
            decoder_outputs = self.decoder_stack(
                decoder_input, cache.get("encoder_outputs"), self_attention_bias,
                cache.get("encoder_decoder_attention_bias"), cache)
            logits = self.decoder_softmax_layer.linear(decoder_outputs)
            logits = tf.reshape(logits, [-1, self.params.target_vocab_size])
            return logits, cache
        return symbols_to_logits_fn


class Discriminator(Transformer):
    def __init__(self, params, is_train):
        super(Discriminator, self).__init__(params, is_train)
        self.init_embed("Discriminator")
        self.out = tf.layers.Dense(units=2, activation=tf.nn.relu)

    def get_loss(self, origin_inputs, targets):
        with tf.variable_scope("Discriminator", initializer=self._initializer, reuse=tf.AUTO_REUSE):
            attention_bias = model_utils.get_padding_bias(targets)  # [batch, 1, 1, src_len]
            encoder_outputs = self.encode(targets, attention_bias)  # [batch, src_len, hidden_size]
            logits = self.decode(origin_inputs, encoder_outputs, attention_bias)
            xentropy, weights = metrics.padded_cross_entropy_loss(
                logits, origin_inputs, self.params.label_smoothing,
                self.params.target_vocab_size)  # [batch, origin_length]
            self.loss = tf.reduce_sum(xentropy, axis=1) / tf.reduce_sum(weights, axis=1) # [batch]
            #prediction = self.argmax_predict(encoder_outputs, attention_bias) # [batch, max_len]
            return tf.reshape(self.loss, (-1, 1))  # [batch, 1]

    def d_loss(self, origin_inputs, targets, labels, is_training=True):
        # self.loss = self.get_loss(origin_inputs, targets)  # [batch, 1]
        self.labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        decode_loss = self.get_loss(origin_inputs, targets)
        # if is_training:
        #     decode_loss = tf.nn.dropout(decode_loss, keep_prob=self.params.relu_dropout)
        self.logits = self.out(decode_loss)  # [batch, 2]
        predictions = tf.cast(tf.argmax(self.logits, axis=-1, name="predict"), tf.int32) # [batch]
        print("predictions", predictions)
        print("labels", labels)
        if not is_training:
            return predictions, decode_loss
        acc = tf.cast(tf.equal(predictions, labels), tf.float32)
        self.accuracy = tf.reduce_mean(acc, name="accuracy")
        d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        # d_loss = tf.reduce_mean(d_loss)
        return d_loss, decode_loss, self.accuracy


def get_state_shape_invariants(tensor):
    """Returns the shape of the tensor but sets middle dims to None."""
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)


if __name__ == "__main__":
    import os

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # tf.enable_eager_execution()
    x_inputs = tf.constant([[1, 2, 3, 0, 0], [3, 4, 5, 6, 0]], dtype=tf.int32)
    y_target = tf.constant([[5, 6, 7, 8, 20], [7, 3, 2, 6, 5]], dtype=tf.int32)
    params = model_params.TransformerBaseParams()
    model = Transformer(params, ModeKeys.TRAIN == "train")
    dis_model = Discriminator(params, ModeKeys.TRAIN == "train")

    # test generator
    gen_targets = model.build_pretrain(x_inputs, targets=None)
    print(gen_targets)
    batch_size = tf.shape(gen_targets)[0]
    f_labels = tf.zeros((batch_size), dtype=tf.int32)
    print("f_labels", f_labels)
    g_loss, decode_loss, acc = dis_model.d_loss(x_inputs, gen_targets, labels=f_labels)
    print(g_loss)
    print(decode_loss)
    print(acc)
    dis_vars = []
    for var in tf.trainable_variables():
        if "Discriminator" in var.name:
            dis_vars.append(var)
            print(var)
    print(len(tf.trainable_variables()))
    print(len(dis_vars))

