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
        if params.shared_embedding_softmax_weights:
            print("sharing embedding!!!")
            self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
                params.vocab_size, params.hidden_size)
            self.encoder_embedding_layer = self.embedding_softmax_layer
            self.decoder_embedding_layer = self.embedding_softmax_layer
            self.decoder_softmax_layer = self.embedding_softmax_layer
        else:
            print("not sharing embedding!!!")
            self.encoder_embedding_layer = embedding_layer.EmbeddingWeights(
                params.source_vocab_size, params.hidden_size, "source_embedding")
            self.decoder_embedding_layer = embedding_layer.EmbeddingWeights(
                params.target_vocab_size, params.hidden_size, "target_embedding")
            self.decoder_softmax_layer = embedding_layer.EmbeddingWeights(
                params.target_vocab_size, params.hidden_size, 'soft_max')
        # done
        self.encoder_stack = EncoderDecoder.EncoderStack(params, is_train, self.mode)
        self.decoder_stack = EncoderDecoder.DecoderStack(params, is_train, self.mode)
        self._initializer = tf.variance_scaling_initializer(
            self.params.initializer_gain, mode="fan_avg", distribution="uniform")

    def build_pretrain(self, inputs, targets):
        # initializer = tf.variance_scaling_initializer(
        #     self.params.initializer_gain, mode="fan_avg", distribution="uniform")
        #
        # with tf.variable_scope("Transformer", initializer=initializer, reuse=tf.AUTO_REUSE):
        if ModeKeys.is_predict_one(self.mode):
            attention_bias = None
        else:
            attention_bias = model_utils.get_padding_bias(inputs)  # [batch, 1, 1, src_len]

        encoder_outputs = self.encode(inputs, attention_bias)  # [batch, src_len, hidden_size]

        if self.mode == ModeKeys.PREDICT_ONE_ENCODER:
            fake_decoder_inputs = tf.zeros([1, 0, self.params.hidden_size])
            fake_decoder_outputs = self.decoder_stack(fake_decoder_inputs, encoder_outputs, None, None, None)

        if targets is None:
            prediction, _ = self.argmax_predict(encoder_outputs, attention_bias)
            return prediction
        else:
            logits = self.decode(targets, encoder_outputs, attention_bias)  # [batch, tgt_len, vocab_size]
            return logits

    def build_generator(self, inputs):
        if ModeKeys.is_predict_one(self.mode):
            self.attention_bias = None
        else:
            self.attention_bias = model_utils.get_padding_bias(inputs)  # [batch, 1, 1, src_len]
        self.encoder_outputs = self.encode(inputs, self.attention_bias)  # [batch, src_len, hidden_size]
        if self.mode == ModeKeys.PREDICT_ONE_ENCODER:
            fake_decoder_inputs = tf.zeros([1, 0, self.params.hidden_size])
            fake_decoder_outputs = self.decoder_stack(fake_decoder_inputs, self.encoder_outputs, None, None, None)

        if self.is_train:
            # if self.mode == tf.estimator.ModeKeys.TRAIN:
            tf.logging.info("!!!!!! using rl predict in traning !!!!!!")
            decoded_ids, decoded_logits, log_probs = self.rl_predict(self.encoder_outputs, self.attention_bias)
            return decoded_ids, decoded_logits, log_probs
        else:
            tf.logging.info("!!!!!!! using argmax_predict in prediction/evaluation !!!!!!!!")
            decoded_ids, decoded_logits =  self.argmax_predict(self.encoder_outputs, self.attention_bias)
            return decoded_ids, decoded_logits, _

    # def build_teacher_force_rollout_generator(self, gen_targets, encoder_outputs, given_num):
    #     """
    #     :param encoder_outputs:  [batch, src_len, enc_size]
    #     :param gen_targets:      [batch, gen_tgt_len]
    #     :param encoder_decoder_attention_bias:  [batch,1,1 src_len]
    #     :return:
    #     """
    #     roll_logits = self.decode(gen_targets, encoder_outputs, self.attention_bias)  # [batch, gen_tgt_len, vocab_size]
    #     roll_samples = tf.multinomial(tf.reshape(roll_logits, [-1, self.params.target_vocab_size]),
    #                                   num_samples=1, output_dtype=tf.int32)  # [batch * gen_tgt_len, vocab_size]
    #     batch_size = tf.shape(encoder_outputs)[0]
    #     roll_samples = tf.reshape(roll_samples, [batch_size, -1])
    #     decoder_ids = tf.concat([gen_targets[:, :given_num], roll_samples[:, given_num:]], axis=1)  # [batch, tgt_len]
    #     return decoder_ids

    def build_new_rollout_generator(self, gen_targets, encoder_outputs, given_num, roll_steps=2):
        """
        :param gen_targets: [batch, src_len, enc_size]
        :param encoder_outputs: [batch, gen_tgt_len]
        :param given_num: [batch,1,1 src_len]
        :return:
        """
        gen_tgt_len = tf.shape(gen_targets)[1]
        roll_gen_tgt = gen_targets[:, :given_num + 1]

        def inner_loop(i, given_num, roll_gen_tgt):
            roll_logits = self.decode(roll_gen_tgt, encoder_outputs,
                                      self.attention_bias)  # [batch, given_num + 1, vocab_size]
            roll_one_step = tf.multinomial(roll_logits[:, given_num, :], num_samples=1,
                                           output_dtype=tf.int32)  # [batch, 1]
            roll_gen_tgt = tf.concat([roll_gen_tgt, roll_one_step], axis=1)  # [batch, tgt_len]
            return i + 1, given_num + 1, roll_gen_tgt

        _, given_num, roll_gen_tgt = tf.while_loop(
            cond=lambda i, given_num, _: (i < roll_steps) & (given_num < gen_tgt_len),
            body=inner_loop,
            loop_vars=[tf.constant(0), given_num, roll_gen_tgt],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([None, None])]
        )
        roll_gen_tgt = tf.concat([roll_gen_tgt, gen_targets[:, given_num + 1:]], axis=1)
        roll_logits = self.decode(roll_gen_tgt, encoder_outputs, self.attention_bias)
        roll_samples = tf.multinomial(tf.reshape(roll_logits, [-1, self.params.target_vocab_size]),
                                      num_samples=1, output_dtype=tf.int32)  # [batch * gen_tgt_len, vocab_size]
        batch_size = tf.shape(encoder_outputs)[0]
        roll_samples = tf.reshape(roll_samples, [batch_size, -1])
        decoder_ids = tf.concat([roll_gen_tgt[:, :given_num], roll_samples[:, given_num:]], axis=1)  # [batch, tgt_len]
        return decoder_ids

    #def get_no_teach_real_loss(self, origin_inputs, gen_targets):
    #    teach_attention_bias = model_utils.get_padding_bias(gen_targets)  # [batch, 1, 1, src_len]
    #    teach_encoder_outputs = self.encode(gen_targets, teach_attention_bias)  # [batch, src_len, hidden_size]
    #    _, teach_logits = self.argmax_predict(teach_encoder_outputs, teach_attention_bias)
    #    teach_xentropy, teach_weights = metrics.padded_cross_entropy_loss(
    #        teach_logits, origin_inputs, self.params.label_smoothing, self.params.target_vocab_size)
    #    teach_loss = tf.reduce_sum(teach_xentropy, axis=1) / tf.reduce_sum(teach_weights, axis=1)  # [batch]
    #    tf.identity(teach_loss[:5], "teach_loss")
    #    mean_teach_loss = tf.reduce_mean(teach_loss, name="mean_teach_loss")
    #    tf.summary.scalar("mean_teach_loss", mean_teach_loss)
    #    return teach_loss

    def get_teach_real_loss(self, origin_inputs, origin_target):
        real_logits = self.build_pretrain(inputs=origin_target, targets=origin_inputs)  # [batch, tgt_len, vocab_size]
        real_xentropy, real_weights = metrics.padded_cross_entropy_loss(
            real_logits, origin_inputs, self.params.label_smoothing, self.params.target_vocab_size)
        real_loss = tf.reduce_sum(real_xentropy, axis=1) / tf.reduce_sum(real_weights, axis=1)  # [batch]
        tf.identity(real_loss[:5], "real_loss")
        mean_real_loss = tf.reduce_mean(real_loss, name="mean_real_loss")
        tf.summary.scalar("mean_real_loss", mean_real_loss)
        return real_loss

    def build_no_teacher_discriminator(self, origin_inputs, gen_target, real_loss, margin=1.0):
        fake_attention_bias = model_utils.get_padding_bias(gen_target)  # [batch, 1, 1, src_len]
        fake_encoder_outputs = self.encode(gen_target, fake_attention_bias)  # [batch, src_len, hidden_size]
        _, fake_logits = self.argmax_predict(fake_encoder_outputs, fake_attention_bias)
        fake_xentropy, fake_weights = metrics.padded_cross_entropy_loss(
            fake_logits, origin_inputs, self.params.label_smoothing,
            self.params.target_vocab_size)  # [batch, origin_length]
        fake_loss = tf.reduce_sum(fake_xentropy, axis=1) / tf.reduce_sum(fake_weights, axis=1)
        tf.identity(fake_loss[:5], "fake_loss")

        mean_fake_loss = tf.reduce_mean(fake_loss, name="mean_fake_loss")
        tf.summary.scalar("mean_fake_loss", mean_fake_loss)

        rewards = 1 / tf.maximum(margin, fake_loss / (real_loss + 1e-12) - 1)  # [batch]
        tf.identity(rewards[:5], "rewards")

        mean_wards = tf.reduce_mean(rewards, name="mean_wards")
        tf.summary.scalar("mean_wards", mean_wards)
        return rewards

    def build_teach_force_discriminator(self, origin_inputs, gen_target, real_loss, margin=1):
        fake_logits = self.build_pretrain(inputs=gen_target, targets=origin_inputs)  # [batch, tgt_length, vocab_size]
        fake_xentropy, fake_weights = metrics.padded_cross_entropy_loss(
            fake_logits, origin_inputs, self.params.label_smoothing,
            self.params.target_vocab_size)  # [batch, origin_length]
        fake_loss = tf.reduce_sum(fake_xentropy, axis=1) / tf.reduce_sum(fake_weights, axis=1)
        tf.identity(fake_loss[:5], "fake_loss")
        mean_fake_loss = tf.reduce_mean(fake_loss, name="mean_fake_loss")
        tf.summary.scalar("mean_fake_loss", mean_fake_loss)
        rewards = 1 / tf.maximum(margin, fake_loss / (real_loss + 1e-12) - 1)  # [batch]
        tf.identity(rewards[:5], "rewards")
        mean_wards = tf.reduce_mean(rewards, name="mean_wards")
        tf.summary.scalar("mean_wards", mean_wards)
        return rewards

    def get_reward_paral_1(self, origin_inputs, gen_targets, origin_target, roll_num, margin, log_probs=None):
        real_loss = self.get_teach_real_loss(origin_target, origin_inputs)  # [batch]
        total_loss = 0
        tgt_len = tf.shape(gen_targets)[1]
        given_num = tf.argmax(tf.concat([tf.ones((1,), dtype=tf.float32) * -1e10,
                                         tf.random_normal((tgt_len,), dtype=tf.float32)[1:]], axis=0),
                              output_type=tf.int32)

        for i in range(roll_num):
            roll_samples = self.build_new_rollout_generator(
                gen_targets, self.encoder_outputs, given_num)  # [batch, tgt_len]
            cur_reward = self.build_teach_force_discriminator(
                gen_target=roll_samples,
                origin_inputs=origin_inputs,
                real_loss=real_loss,
                margin=margin)
            tf.identity(given_num, "given_num")
            total_loss += log_probs[:, given_num] * tf.stop_gradient(cur_reward)
            tf.identity(log_probs[:, given_num][:5], "log_probs")
        return - tf.reduce_mean(total_loss) / roll_num, tf.reduce_mean(real_loss)

    def get_reward_paral_2(self, origin_inputs, gen_targets, origin_target, roll_num, margin, log_probs):
        real_loss = self.get_teach_real_loss(origin_target, origin_inputs)  # [batch]
        total_loss = 0
        tgt_len = tf.shape(gen_targets)[1]
        given_num = tf.argmax(tf.concat([tf.ones((1,), dtype=tf.float32) * -1e10,
                                         tf.random_normal((tgt_len,), dtype=tf.float32)[1:]], axis=0),
                              output_type=tf.int32)
        for i in range(roll_num):
            roll_samples = self.build_new_rollout_generator(
                gen_targets, self.encoder_outputs, given_num)  # [batch, tgt_len]
            cur_reward = self.build_no_teacher_discriminator(
                gen_target=roll_samples,
                origin_inputs=origin_inputs,
                real_loss=real_loss,
                margin=margin)
            tf.identity(given_num, "given_num")

            total_loss += log_probs[:, given_num] * tf.stop_gradient(cur_reward)
            tf.identity(log_probs[:, given_num][:5], "log_probs")
        return - tf.reduce_mean(total_loss) / roll_num, tf.reduce_mean(real_loss)

    def get_reward_mono(self, origin_inputs, gen_targets, roll_num, margin, log_probs):
        teach_loss = self.get_teach_real_loss(origin_inputs, gen_targets)
        total_loss = 0
        tgt_len = tf.shape(gen_targets)[1]
        given_num = tf.argmax(tf.concat([tf.ones((1,), dtype=tf.float32) * -1e10,
                                         tf.random_normal((tgt_len,), dtype=tf.float32)[1:]], axis=0),
                              output_type=tf.int32)
        for i in range(roll_num):
            roll_samples = self.build_new_rollout_generator(
                gen_targets, self.encoder_outputs, given_num)  # [batch, tgt_len]
            cur_reward = self.build_no_teacher_discriminator(
                gen_target=roll_samples,
                origin_inputs=origin_inputs,
                real_loss=teach_loss,
                margin=margin)
            tf.identity(given_num, "given_num")

            total_loss += log_probs[:, given_num] * tf.stop_gradient(cur_reward)
            tf.identity(log_probs[:, given_num][:5], "log_probs")
        return - tf.reduce_mean(total_loss) / roll_num, tf.reduce_mean(teach_loss)

    def encode(self, inputs, attention_bias):
        """Generate continuous representation for inputs.

        Args:
          inputs: int tensor with shape [batch_size, input_length].
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

        Returns:
          float tensor with shape [batch_size, input_length, hidden_size]
        """
        with tf.name_scope("encode"):
            # Prepare inputs to the layer stack by adding positional encodings and
            # applying dropout.
            embedded_inputs = self.encoder_embedding_layer(inputs, not ModeKeys.is_predict_one(self.mode))
            if ModeKeys.is_predict_one(self.mode):
                inputs_padding = None
            else:
                inputs_padding = model_utils.get_padding(inputs)

            # add_pos_encoding
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
        """Generate logits for each value in the target sequence.

        Args:
          targets: target values for the output sequence.
            int tensor with shape [batch_size, target_length]
          encoder_outputs: continuous representation of input sequence.
            float tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

        Returns:
          float32 tensor with shape [batch_size, target_length, vocab_size]
        """
        with tf.name_scope("decode"):
            # Prepare inputs to decoder layers by shifting targets, adding positional
            # encoding and applying dropout.
            # !!!
            # decoder_inputs = self.embedding_softmax_layer(targets)
            decoder_inputs = self.decoder_embedding_layer(targets, not ModeKeys.is_predict_one(self.mode))
            # done
            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
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
            # !!!
            # logits = self.embedding_softmax_layer.linear(outputs)
            logits = self.decoder_softmax_layer.linear(outputs)
            # done
            return logits

    def argmax_predict(self, encoder_outputs, encoder_decoder_attention_bias):
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
            pass
        else:
            def inner_loop(i, finished, next_id, decoded_ids, decoded_logits, cache):
                """One step of greedy decoding."""
                logits, cache = symbols_to_logits_fn(next_id, i, cache)  # [batch, vocab_size]
                next_id = tf.argmax(logits, -1, output_type=tf.int32)
                finished |= tf.equal(next_id, EOS_ID)
                next_id = tf.reshape(next_id, shape=[-1, 1])
                decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
                logits = tf.expand_dims(logits, axis=1)
                decoded_logits = tf.concat([decoded_logits, logits], axis=1)
                return i + 1, finished, next_id, decoded_ids, decoded_logits, cache

            def is_not_finished(i, finished, _1, _2, _3, _4):
                return (i < max_decode_length) & tf.logical_not(tf.reduce_all(finished))

            decoded_logits = tf.zeros([batch_size, 0, self.params.target_vocab_size], dtype=tf.float32)
            decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int32)
            finished = tf.fill([batch_size], False)
            next_id = tf.zeros([batch_size, 1], dtype=tf.int32)

            _, _, _, decoded_ids, decoded_logits, _ = tf.while_loop(
                cond=is_not_finished,
                body=inner_loop,
                loop_vars=[tf.constant(0), finished, next_id, decoded_ids, decoded_logits, cache],
                shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None, None]),
                    nest.map_structure(get_state_shape_invariants, cache),
                ])

            return decoded_ids, decoded_logits
    

    def rl_predict_new(self, encoder_outputs, encoder_decoder_attention_bias):
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
            pass
        else:
            def inner_loop(i, finished, next_id, decoded_ids, log_probs, cache):
                # print("time step:", i)
                """One step of greedy decoding."""
                prev_id = next_id
                print("prev_id", prev_id)
                logits, cache = symbols_to_logits_fn(next_id, i, cache)
                categorical = tf.contrib.distributions.Categorical(logits=logits)
                next_id = categorical.sample()
                log_prob = categorical.log_prob(next_id)  # [batch,]
                finished |= tf.equal(next_id, EOS_ID)
                finished = tf.reshape(finished, (-1,))
                next_id = tf.reshape(next_id, shape=[-1, 1])
                mask = tf.cast(tf.math.not_equal(prev_id, EOS_ID), dtype=tf.int32)
                next_id = next_id * mask
                def pad_fn():
                    print("!!!!right here!!!", i)
                    mask_pad = tf.cast(tf.math.not_equal(prev_id, PAD_ID), dtype=tf.int32)
                    return next_id * mask_pad
                next_id = tf.cond(tf.less(i, 1), lambda: next_id, pad_fn)
                log_prob = tf.reshape(log_prob, shape=[-1, 1])
                decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
                log_probs = tf.concat([log_probs, log_prob], axis=1)  # [batch, len]
                return i + 1, finished, next_id, decoded_ids, log_probs, cache

            def is_not_finished(i, finished, _1, _2, _3, _4):
                return (i < max_decode_length) & tf.logical_not(tf.reduce_all(finished))

            decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int32)
            log_probs = tf.zeros([batch_size, 0], dtype=tf.float32)
            finished = tf.fill([batch_size], False)
            next_id = tf.zeros([batch_size, 1], dtype=tf.int32)

            _, _, _, decoded_ids, log_probs, cache = tf.while_loop(
                cond=is_not_finished,
                body=inner_loop,
                loop_vars=[tf.constant(0), finished, next_id, decoded_ids, log_probs, cache],
                shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    nest.map_structure(get_state_shape_invariants, cache),
                ])

            # return {"outputs": decoded_ids, "scores": tf.ones([batch_size, 1])}
            return decoded_ids, log_probs


    def rl_predict(self, encoder_outputs, encoder_decoder_attention_bias):
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
            pass
        else:
            def inner_loop(i, finished, next_id, decoded_ids, log_probs, decoded_logits, cache):
                # print("time step:", i)
                """One step of greedy decoding."""

                logits, cache = symbols_to_logits_fn(next_id, i, cache)
                categorical = tf.contrib.distributions.Categorical(logits=logits)
                next_id = categorical.sample()
                log_prob = categorical.log_prob(next_id)  # [batch,]
                finished |= tf.equal(next_id, EOS_ID)
                finished = tf.reshape(finished, (-1,))
                next_id = tf.reshape(next_id, shape=[-1, 1])
                log_prob = tf.reshape(log_prob, shape=[-1, 1])
                decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
                log_probs = tf.concat([log_probs, log_prob], axis=1)  # [batch, len]
                logits = tf.expand_dims(logits, axis=1)
                decoded_logits = tf.concat([decoded_logits, logits], axis=1)
                return i + 1, finished, next_id, decoded_ids, log_probs, decoded_logits, cache

            def is_not_finished(i, finished, _1, _2, _3, _4, _5):
                return (i < max_decode_length) & tf.logical_not(tf.reduce_all(finished))

            decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int32)
            log_probs = tf.zeros([batch_size, 0], dtype=tf.float32)
            decoded_logits = tf.zeros([batch_size, 0, self.params.target_vocab_size], dtype=tf.float32)
            finished = tf.fill([batch_size], False)
            next_id = tf.zeros([batch_size, 1], dtype=tf.int32)

            _, _, _, decoded_ids, log_probs, decoded_logits, cache = tf.while_loop(
                cond=is_not_finished,
                body=inner_loop,
                loop_vars=[tf.constant(0), finished, next_id, decoded_ids, log_probs, decoded_logits, cache],
                shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None, None]),
                    nest.map_structure(get_state_shape_invariants, cache),
                ])

            # return {"outputs": decoded_ids, "scores": tf.ones([batch_size, 1])}
            return decoded_ids, decoded_logits, log_probs

    def _get_symbols_to_logits_fn(self, max_decode_length):
        """Returns a decoding function that calculates logits of the next tokens."""
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
            """Generate logits for next potential IDs.

            Args:
              ids: Current decoded sequences.
                int tensor with shape [batch_size * beam_size, i + 1]
              i: Loop index
              cache: dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.

            Returns:
              Tuple of
                (logits with shape [batch_size * beam_size, vocab_size],
                 updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]  # [batch, 1]

            # decoder_input = ids[:, :]     # [batch, 1]
            # print("decoder_input:", decoder_input.shape)

            # Preprocess decoder input by getting embeddings and adding timing signal.
            # !!!!!!!!
            decoder_input = self.decoder_embedding_layer(decoder_input, not ModeKeys.is_predict_one(
                self.mode))  # [batch, 1, hidden_size]
            # !!!!!!!!
            if ModeKeys.is_predict_one(self.mode):
                decoder_input = decoder_input * (1 - tf.to_float(tf.equal(i, 0)))

            # add position embedding
            # decoder_input += timing_signal[i:i + 1]
            slice_pos_encoding = tf.slice(timing_signal, [i, 0], [1, self.params.hidden_size],
                                          name='slice_pos_encoding')  # [1, hidden_size]
            decoder_input += slice_pos_encoding

            if decoder_self_attention_bias is None:
                self_attention_bias = None
            else:
                self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]  # [1, 1, 1, time_step]
                # self_attention_bias = decoder_self_attention_bias[:, :, :i+1, :i+1] # [1, 1, 1, time_step]
            # print("attention bias:", self_attention_bias.shape)
            decoder_outputs = self.decoder_stack(
                decoder_input, cache.get("encoder_outputs"), self_attention_bias,
                cache.get("encoder_decoder_attention_bias"), cache)
            logits = self.decoder_softmax_layer.linear(decoder_outputs)
            # logits = tf.squeeze(logits, axis=[1])
            logits = tf.reshape(logits, [-1, self.params.target_vocab_size])
            return logits, cache

        return symbols_to_logits_fn


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
    y_target = tf.constant([[5, 6, 7, 8, 20], [7, 3, 2, 6, 5]], dtype=tf.int32)
    params = model_params.TransformerBaseParams()
    model = Transformer(params, ModeKeys.TRAIN == "train")

    # test pretrain
    pretrain_out = model.build_pretrain(x_inputs, y_target)
    # test generator
    gen_targets, log_probs = model.build_generator(x_inputs)
    # real_loss = model.get_teach_real_loss(origin_target=y_target, origin_inputs=x_inputs)
    # reward = model.build_no_teacher_discriminator(origin_inputs=x_inputs, gen_target=gen_target, real_loss=real_loss,
    #                                               margin=0.1)
    # roll_samples = model.build_new_rollout_generator(gen_target, model.encoder_outputs, given_num=3)
    # rewards_1 = model.get_reward_teacher_force(origin_inputs=x_inputs,
    #                                            gen_targets=gen_target,
    #                                            origin_target=y_target,
    #                                            roll_num=2,
    #                                            margin=1,
    #                                            log_probs=log_probs)

    # rewards_2 = model.get_reward_paral_2(origin_inputs=x_inputs,
    #                                      gen_targets=gen_targets,
    #                                      origin_target=y_target,
    #                                      roll_num=2,
    #                                      margin=1,
    #                                      log_probs=log_probs)

    rewards_3 = model.get_reward_mono(origin_inputs=x_inputs,
                                      gen_targets=gen_targets,
                                      roll_num=2,
                                      margin=1,
                                      log_probs=log_probs)
    print(rewards_3)
