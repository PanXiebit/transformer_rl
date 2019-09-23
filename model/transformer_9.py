from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import nest
from model.base import ModeKeys
from model import embedding_layer
from model import model_utils
from model import model_params
from model import beam_search
from model.EncoderDecoder import EncoderStack, DecoderStack
from utils import metrics
from utils.tokenizer import EOS_ID, PAD_ID
import time

class Transformer(object):
    """Transformer model for sequence to sequence data.

    Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

    The Transformer model consists of an encoder and decoder. The input is an int
    sequence (or a batch of sequences). The encoder produces a continous
    representation, and the decoder uses the encoder output to generate
    probabilities for the output sequence.
    """


    def __init__(self, params, is_train, mode=None, scope=None):
        """Initialize layers to build Transformer model.

        Args:
          params: hyperparameter object defining layer sizes, dropout values, etc.
          is_train: boolean indicating whether the model is in training mode. Used to
            determine if dropout layers should be added.
        """
        self.dropout_rate = tf.placeholder_with_default(0.0, shape=[], name="dropout_rate")

        self.is_train = is_train
        self.params = params
        self.name_scope = scope

        # reset dropout rate using placeholder,
        # when inference, the dropout_rate is 0.0, when training is 0.1
        self.params.layer_postprocess_dropout = self.dropout_rate
        self.params.attention_dropout = self.dropout_rate
        self.params.relu_dropout = self.dropout_rate

        if mode is not None:
            self.mode = mode
        elif self.is_train:
            self.mode = ModeKeys.TRAIN
        else:
            self.mode = ModeKeys.PREDICT

        self.initializer = tf.variance_scaling_initializer(
            self.params.initializer_gain, mode="fan_avg", distribution="uniform")
        # done
        self.encoder_stack = EncoderStack(params, is_train, self.mode)
        self.decoder_stack = DecoderStack(params, is_train, self.mode)

        with tf.variable_scope(self.name_scope):
            if params.shared_embedding_softmax_weights:
                self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
                    params.vocab_size, params.hidden_size)
                self.encoder_embedding_layer = self.embedding_softmax_layer
                self.decoder_embedding_layer = self.embedding_softmax_layer
                self.decoder_softmax_layer = self.embedding_softmax_layer
            else:
                self.encoder_embedding_layer = embedding_layer.EmbeddingWeights(
                    params.source_vocab_size, params.hidden_size, "source_embedding")
                self.decoder_embedding_layer = embedding_layer.EmbeddingWeights(
                    params.target_vocab_size, params.hidden_size, "target_embedding")
                self.decoder_softmax_layer = embedding_layer.EmbeddingWeights(
                    params.target_vocab_size, params.hidden_size, 'sot_max')



    def inference(self, inputs, targets=None, reuse=None):
        with tf.variable_scope(self.name_scope, initializer=self.initializer, reuse=reuse):

            if ModeKeys.is_predict_one(self.mode):
                attention_bias = None
            else:
                attention_bias = model_utils.get_padding_bias(inputs)
            encoder_outputs = self.encode(inputs, attention_bias)
            if self.mode == ModeKeys.PREDICT_ONE_ENCODER:
                fake_decoder_inputs = tf.zeros([1, 0, self.params.hidden_size])
                fake_decoder_outputs = self.decoder_stack(fake_decoder_inputs, encoder_outputs, None, None, None)
            if targets is None:
                return self.predict(encoder_outputs, attention_bias)
            else:
                logits = self.decode(targets, encoder_outputs, attention_bias)
                return logits


    def encode(self, inputs, attention_bias):
        with tf.name_scope("encode"):
            embedded_inputs = self.encoder_embedding_layer(inputs, not ModeKeys.is_predict_one(self.mode))
            if ModeKeys.is_predict_one(self.mode):
                inputs_padding = None
            else:
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
        with tf.name_scope("decode"):
            decoder_inputs = self.decoder_embedding_layer(targets, not ModeKeys.is_predict_one(self.mode))
            # done
            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
                decoder_inputs = tf.pad(
                    decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
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
            # !!!
            # logits = self.embedding_softmax_layer.linear(outputs)
            logits = self.decoder_softmax_layer.linear(outputs)
            # done
            return logits


    def _get_symbols_to_logits_fn(self, max_decode_length):
        if ModeKeys.is_predict_one(self.mode):
            timing_signal = model_utils.get_position_encoding(
                self.params.max_length, self.params.hidden_size
            )
            timing_signal = tf.slice(timing_signal, [0, 0], [max_decode_length + 1, self.params.hidden_size],
                                     name='slice_timing_signal')
        else:
            timing_signal = model_utils.get_position_encoding(
                max_decode_length + 1, self.params.hidden_size)

        if ModeKeys.is_predict_one(self.mode):
            decoder_self_attention_bias = None
        else:
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            decoder_input = ids[:, -1:]
            decoder_input = self.decoder_embedding_layer(decoder_input, not ModeKeys.is_predict_one(self.mode))
            # !!!!!!!!
            if ModeKeys.is_predict_one(self.mode):
                decoder_input = decoder_input * (1 - tf.to_float(tf.equal(i, 0)))

            # decoder_input += timing_signal[i:i + 1]
            slice_pos_encoding = tf.slice(timing_signal, [i, 0], [1, self.params.hidden_size], name='slice_pos_encoding')
            decoder_input += slice_pos_encoding

            if decoder_self_attention_bias is None:
                self_attention_bias = None
            else:
                self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_outputs = self.decoder_stack(
                decoder_input, cache.get("encoder_outputs"), self_attention_bias,
                cache.get("encoder_decoder_attention_bias"), cache)
            logits = self.decoder_softmax_layer.linear(decoder_outputs)
            # logits = tf.squeeze(logits, axis=[1])
            logits = tf.reshape(logits, [-1, self.params.target_vocab_size])
            return logits, cache

        return symbols_to_logits_fn


    def predict(self, encoder_outputs, encoder_decoder_attention_bias):
        """Return predicted sequence."""
        if ModeKeys.is_predict_one(self.mode):
            batch_size = 1
        else:
            batch_size = tf.shape(encoder_outputs)[0]
        input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = input_length + self.params.extra_decode_length

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)

        # Create cache storing decoder attention values for each layer.
        cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.params.hidden_size]),
                "v": tf.zeros([batch_size, 0, self.params.hidden_size]),
            } for layer in range(self.params.num_hidden_layers)}

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs
        if not ModeKeys.is_predict_one(self.mode):
            cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

        if self.params.beam_size > 1:
            print("!!!!!!!!!!! right here, beam_size = %i!!!!!!!!!!!!"%self.params.beam_size)
            # Use beam search to find the top beam_size sequences and scores.
            decoded_ids, scores = beam_search.sequence_beam_search(
                symbols_to_logits_fn=symbols_to_logits_fn,
                initial_ids=initial_ids,
                initial_cache=cache,
                vocab_size=self.params.target_vocab_size,
                beam_size=self.params.beam_size,
                alpha=self.params.alpha,
                max_decode_length=max_decode_length,
                eos_id=EOS_ID)

            # Get the top sequence for each batch element
            top_decoded_ids = decoded_ids[:, 0, 1:]
            top_scores = scores[:, 0]

            return {"outputs": top_decoded_ids, "scores": top_scores}

        else:

            def inner_loop(i, finished, next_id, decoded_ids, cache):
                """One step of greedy decoding."""
                logits, cache = symbols_to_logits_fn(next_id, i, cache)
                next_id = tf.argmax(logits, -1, output_type=tf.int32)
                finished |= tf.equal(next_id, EOS_ID)
                # next_id = tf.expand_dims(next_id, axis=1)
                next_id = tf.reshape(next_id, shape=[-1, 1])
                decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
                return i + 1, finished, next_id, decoded_ids, cache

            def is_not_finished(i, finished, *_):
                return (i < max_decode_length) & tf.logical_not(tf.reduce_all(finished))

            decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int32)
            finished = tf.fill([batch_size], False)
            next_id = tf.zeros([batch_size, 1], dtype=tf.int32)
            _, _, _, decoded_ids, _ = tf.while_loop(
                is_not_finished,
                inner_loop,
                [tf.constant(0), finished, next_id, decoded_ids, cache],
                shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    nest.map_structure(get_state_shape_invariants, cache),
                ])

            return {"outputs": decoded_ids, "scores": tf.ones([batch_size, 1])}


    def decoder_predict(self, pos_idx, pre_id, cache):
        input_length = tf.shape(cache['encoder_outputs'])[1]
        max_decode_length = input_length + self.params.extra_decode_length
        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

        # One step of greedy decoding.
        logits, cache = symbols_to_logits_fn(pre_id, pos_idx, cache)
        next_id = tf.argmax(logits, -1, output_type=tf.int32)
        next_id = tf.reshape(next_id, shape=[-1, 1])

        return next_id, cache


    def call_decoder_predict(self, pos_idx, pre_id, cache):
        # just for build same name scope with training
        initializer = tf.variance_scaling_initializer(
            self.params.initializer_gain, mode="fan_avg", distribution="uniform")
        with tf.variable_scope("Transformer", initializer=initializer):
            fake_inputs = tf.placeholder(dtype=tf.int32, shape=[self.params.batch_size, None])
            _ = self.encode(fake_inputs, None)

            return self.decoder_predict(pos_idx, pre_id, cache)


def get_state_shape_invariants(tensor):
    """Returns the shape of the tensor but sets middle dims to None."""
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)



if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.enable_eager_execution()
    x_inputs = tf.constant([[1, 2, 3, 0, 0], [3, 4, 5, 6, 0]], dtype=tf.int32)
    y_target = tf.constant([[1, 2, 3, 0, 0], [3, 4, 5, 6, 0]], dtype=tf.int32)
    params = model_params.TransformerBaseParams()
    model = Transformer(params, ModeKeys.TRAIN == "train", scope="Transformer")

    # test pretrain
    pretrain_out = model.inference(x_inputs, y_target)
    for var in tf.global_variables():
        print(var)
    # print(len(tf.global_variables()))

