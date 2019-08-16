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
        
        with tf.device('/cpu:0'):
           self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_pl")
           self.params.layer_postprocess_dropout = self.dropout_pl
           self.params.attention_dropout = self.dropout_pl
           self.relu_dropout = self.dropout_pl

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

    def build_pretrain_mono(self, inputs, targets):
        inputs_length = tf.reduce_sum(tf.cast(tf.not_equal(inputs, EOS_ID), tf.int32), axis=1) # [batch]
        print("inputs_length", inputs_length)
        max_len = inputs_length[tf.argmax(inputs_length)]
        print("max_len", max_len)
        batch_size = tf.shape(inputs)[0]

        pad_inputs = tf.zeros([0, max_len], dtype=tf.int32)
        def inner_loop(i, pad_inputs):
            ori_length = inputs_length[i]
            ori_input = tf.reshape(inputs[i][:ori_length], [1, -1])
            pad_input = tf.pad(ori_input, [[0,0], [0, max_len - ori_length]])
            pad_inputs = tf.concat([pad_inputs, pad_input], axis=0)   
            print("pad_inputs", pad_inputs)
            return i + 1, pad_inputs
        _, pad_inputs = tf.while_loop(
            cond=lambda i,_: i < batch_size,
            body=inner_loop,
            loop_vars=[tf.constant(0), pad_inputs],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, None])]
        )
        with tf.variable_scope("Transformer", initializer=self._initializer, reuse=tf.AUTO_REUSE):
            attention_bias = model_utils.get_padding_bias(pad_inputs)  # [batch, 1, 1, src_len]
            encoder_outputs = self.encode(pad_inputs, attention_bias)  # [batch, src_len, hidden_size]
            #encoder_outputs = tf.stop_gradient(encoder_outputs)
            if targets is None:
                prediction = self.argmax_predict(encoder_outputs, attention_bias)
                return prediction
            else:
                tf.logging.info("!!!!!!!!!! pretrain decoder !!!!!!!!!!!!!!!!!!")
                logits = self.decode(targets, encoder_outputs, attention_bias)  # [batch, tgt_len, vocab_size]
                return logits

    def build_generator(self, inputs):
        with tf.variable_scope("Transformer", initializer=self._initializer, reuse=tf.AUTO_REUSE):
            self.attention_bias = model_utils.get_padding_bias(inputs)  # [batch, 1, 1, src_len]
            self.encoder_outputs = self.encode(inputs, self.attention_bias)  # [batch, src_len, hidden_size]
            tf.logging.info("!!!!!!! using argmax_predict in generator !!!!!!!!")
            decoded_ids = self.argmax_predict(self.encoder_outputs, self.attention_bias)
            return decoded_ids

    def build_padding_rollout_generator(self, gen_samples, max_len, given_num):
        with tf.variable_scope("Transformer", initializer=self._initializer, reuse=tf.AUTO_REUSE):
            encoder_outputs = self.encoder_outputs
            def condition(given_num, _):
                return given_num < max_len
            def inner_loop(given_num, given_y):
                logits = self.decode(given_y, encoder_outputs, self.attention_bias)
                next_logits = logits[:, given_num, :]  # [batch, decoder_vocab_size]
                next_probs = tf.nn.softmax(next_logits)
                log_probs = tf.log(next_probs)
                next_sample = tf.multinomial(log_probs, num_samples=1)
                next_sample = tf.cast(next_sample, dtype=tf.int32)
                given_y = tf.concat([given_y[:, :given_num], next_sample], axis=1)
                given_y = tf.pad(given_y, [[0, 0], [0, max_len - given_num - 1]])
                return given_num + 1, given_y

            given_y = gen_samples[:, :given_num]
            init_given_y = tf.pad(given_y, [[0, 0], [0, max_len - given_num]])
            init_given_num = given_num

            given_num, roll_sample = tf.while_loop(
                cond=condition,
                body=inner_loop,
                loop_vars=[init_given_num, init_given_y],
                shape_invariants=[init_given_num.get_shape(),
                                  tf.TensorShape([None, None])]
            )
            return roll_sample

    def get_reward(self, origin_inputs, gen_targets, roll_num):
        max_len = tf.shape(gen_targets)[1]
        batch_size = tf.shape(gen_targets)[0]
        gen_targets_mask = tf.cast(tf.not_equal(gen_targets, PAD_ID), tf.float32)
        total_rewards = []
        for i in range(roll_num):
            tf.logging.info("roll_num: {}".format(i))
            start_time = time.time()

            given_num = tf.constant(1)
            rewards = tf.zeros((batch_size, 0), dtype=tf.float32)

            def inner_loop(given_num, rewards):
                tf.logging.info("given_num {}".format(given_num))
                roll_sample = self.build_padding_rollout_generator(
                    gen_samples=gen_targets,
                    max_len=max_len,
                    given_num=given_num
                )
                cur_reward = self.build_bleu_discriminator(
                    origin_inputs=origin_inputs,
                    gen_targets=roll_sample
                )
                rewards = tf.concat([rewards, cur_reward], axis=1)
                return given_num + 1, rewards

            _, rewards = tf.while_loop(
                cond=lambda given_num, _: tf.less(given_num, max_len),
                body=inner_loop,
                loop_vars=[given_num, rewards],
                shape_invariants=[tf.TensorShape([]),
                                  tf.TensorShape([None, None])]
            )   # rewards [batch, max_len - 1]
            tf.logging.info("rollout once need time:{}".format(time.time()-start_time))
            total_rewards.append(tf.expand_dims(rewards, axis=1))  # roll_num * [batch, 1, max_len-1]
        total_rewards = tf.concat(total_rewards, axis=1)  # [batch, roll_num, max_len-1]
        total_rewards = tf.reduce_mean(total_rewards, axis=1, keepdims=False)  # [batch, max_len-1]
        cur_reward = self.build_bleu_discriminator(origin_inputs=origin_inputs, gen_targets=gen_targets)
        total_rewards = tf.concat([total_rewards, cur_reward], axis=1) # [batch, max_len]
        total_rewards *= gen_targets_mask
        return total_rewards

    def get_one_reward(self, origin_inputs, gen_targets, roll_num):
        max_len = tf.shape(gen_targets)[1]
        lengths = tf.reduce_sum(tf.cast(tf.not_equal(gen_targets, PAD_ID), tf.int32), axis=1)
        min_len = lengths[tf.argmin(lengths)]
        gen_targets_mask = tf.cast(tf.not_equal(gen_targets, PAD_ID), tf.float32)

        given_num = tf.argmax(
            tf.concat([tf.ones((1,), dtype=tf.float32) * (-1e5), tf.random_normal([min_len-1,], dtype=tf.float32)], 
                      axis=0), output_type=tf.int32)
        print("given_num", given_num)
        total_rewards = []
        for i in range(roll_num):
            tf.logging.info("roll_num: {}".format(i))
            roll_sample = self.build_padding_rollout_generator(
                gen_samples=gen_targets,
                max_len=max_len,
                given_num=given_num
            )
            print("roll_sample", roll_sample)
            cur_reward = self.build_bleu_discriminator(
                origin_inputs=origin_inputs,
                gen_targets=roll_sample
            )   # [batch ,1]
            total_rewards.append(cur_reward) # list, [batch,1] * roll_num
        total_rewards = tf.reduce_mean(tf.concat(total_rewards, axis=1), axis=1) # [bacth, roll_num] -> [batch, 1]
        return given_num, total_rewards

    def get_one_reward_baseline(self, origin_inputs, gen_targets, roll_num):
        baseline = self.build_bleu_discriminator(origin_inputs=origin_inputs,
                                                 gen_targets=gen_targets)
        tf.identity(baseline[:5], "baseline")
        max_len = tf.shape(gen_targets)[1]
        lengths = tf.reduce_sum(tf.cast(tf.not_equal(gen_targets, PAD_ID), tf.int32), axis=1)
        min_len = lengths[tf.argmin(lengths)]
        gen_targets_mask = tf.cast(tf.not_equal(gen_targets, PAD_ID), tf.float32)

        given_num = tf.argmax(
            tf.concat([tf.ones((1,), dtype=tf.float32) * (-1e5), tf.random_normal([min_len-1,], dtype=tf.float32)], 
                      axis=0), output_type=tf.int32)
        print("given_num", given_num)
        total_rewards = []
        for i in range(roll_num):
            tf.logging.info("roll_num: {}".format(i))
            roll_sample = self.build_padding_rollout_generator(
                gen_samples=gen_targets,
                max_len=max_len,
                given_num=given_num
            )
            print("roll_sample", roll_sample)
            cur_reward = self.build_bleu_discriminator(
                origin_inputs=origin_inputs,
                gen_targets=roll_sample
            )   # [batch ,1]
            total_rewards.append(cur_reward) # list, [batch,1] * roll_num
        total_rewards = tf.reduce_mean(tf.concat(total_rewards, axis=1), axis=1) # [bacth, roll_num] -> [batch, ]
        tf.identity(total_rewards[:5], "mean_rewards")
        total_rewards = tf.maximum(0.0, total_rewards - tf.reshape(baseline, [-1])) 
        return given_num, total_rewards
    


    def get_g_loss(self, gen_targets, rewards):
        with tf.variable_scope("Transformer", initializer=self._initializer, reuse=tf.AUTO_REUSE):
            logits = self.decode(targets=gen_targets, encoder_outputs=self.encoder_outputs,
                                 attention_bias=self.attention_bias)
            batch_size = tf.shape(gen_targets)[0]
            probs = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.reduce_sum(
                tf.one_hot(tf.reshape(gen_targets, [-1]), self.params.target_vocab_size, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(probs, [-1, self.params.target_vocab_size]), 1e-20, 1.0)), axis=1)
            rewards = tf.stop_gradient(rewards)
            g_loss = - tf.reduce_sum(log_probs * tf.reshape(rewards, [-1])) / tf.to_float(batch_size)
            return g_loss

    def get_one_g_loss(self, gen_targets, given_num, rewards):
        given_y = gen_targets[:, :given_num]
        print("given_y", given_y)
        with tf.variable_scope("Transformer", initializer=self._initializer, reuse=tf.AUTO_REUSE):
            logits = self.decode(targets=given_y, encoder_outputs=self.encoder_outputs,
                                 attention_bias=self.attention_bias)
            batch_size = tf.shape(gen_targets)[0]
            probs = tf.nn.softmax(logits[:, -1, :], axis=-1) # probability, [batch, dec_vocab_size]
            log_probs = tf.reduce_sum(tf.one_hot(given_y[:, -1], self.params.target_vocab_size, 1.0, 0.0) * tf.log(
                tf.clip_by_value(probs, 1e-20, 1.0)), axis=1)
            rewards = tf.stop_gradient(rewards)
            g_loss = - tf.reduce_sum(log_probs * tf.reshape(rewards, [-1])) / tf.to_float(batch_size)
            return g_loss

    def build_bleu_discriminator(self, origin_inputs, gen_targets):
        with tf.variable_scope("Transformer", initializer=self._initializer, reuse=tf.AUTO_REUSE):
            fake_prediction = self.build_pretrain(gen_targets, targets=None)  # argmax_predict
            fake_bleu = tf.py_func(metrics.compute_bleu_batch, (origin_inputs, fake_prediction), tf.float32)
            tf.identity(fake_bleu[:5], "fake_bleu")
            cur_reward = fake_bleu
            return tf.reshape(cur_reward, (-1, 1))

    def get_real_blue_teach(self, origin_inputs, origin_targets):
        real_logits = self.build_pretrain(origin_targets, targets=origin_inputs)
        real_prediction = tf.argmax(real_logits, axis=-1)  # [batch, ori_inp_len]
        comp_bleu_time = time.time()
        real_bleu = tf.py_func(metrics.compute_bleu_batch, (origin_inputs, real_prediction), tf.float32)
        tf.logging.info("comp_bleu_time: {}".format(time.time() - comp_bleu_time))
        return real_bleu

    def get_fake_blue_teach(self, origin_inputs, gen_targets):
        fake_logits = self.build_pretrain(gen_targets, targets=origin_inputs)  # argmax_predict
        fake_prediction = tf.argmax(fake_logits, axis=-1)  # [batch, ori_inp_len]
        fake_bleu = tf.py_func(metrics.compute_bleu_batch, (origin_inputs, fake_prediction), tf.float32)
        return fake_bleu

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

    def rl_predict_new(self, encoder_outputs, encoder_decoder_attention_bias):
        if ModeKeys.is_predict_one(self.mode):
            batch_size = 1
        else:
            batch_size = tf.shape(encoder_outputs)[0]
        input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = input_length + self.params.extra_decode_length
        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)
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
            return decoded_ids, log_probs

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


def get_state_shape_invariants(tensor):
    """Returns the shape of the tensor but sets middle dims to None."""
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)


if __name__ == "__main__":
    import os
    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    tf.enable_eager_execution()
    x_inputs = tf.constant([[1, 2, 3, 0, 0], [3, 4, 5, 6, 0]], dtype=tf.int32)
    y_target = tf.constant([[5, 6, 7, 8, 20], [7, 3, 2, 6, 5]], dtype=tf.int32)
    params = model_params.TransformerBaseParams()
    model = Transformer(params, ModeKeys.TRAIN == "train")

    # test generator
    gen_targets = model.build_generator(x_inputs)

    # lengths = tf.reduce_sum(tf.cast(tf.not_equal(gen_targets, 0), tf.int32), axis=1)
    # min_len = lengths[tf.math.argmin(lengths)]
    # print(min_len)
    # max_len = tf.shape(gen_targets)[1]
    # roll_samples = model.build_padding_rollout_generator(gen_samples=gen_targets,
    #                                                      max_len=max_len,
    #                                                      given_num=tf.constant(2))
    # print(roll_samples)
    #given_num, total_rewards = model.get_one_reward_baseline(origin_inputs=x_inputs,
    #                                 gen_targets=gen_targets,
    #                                 roll_num=2)
    #print("total_reward", total_rewards)
    #g_loss = model.get_one_g_loss(gen_targets, given_num, total_rewards)
    #print(g_loss)
    print(gen_targets)
    logits = model.build_pretrain_mono(gen_targets, x_inputs)
    print(logits)
