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
import numpy as np
import random
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
            prediction, _, _ = self.argmax_predict(encoder_outputs, attention_bias)
            return prediction
        else:
            tf.logging.info("!!!!!!!!!!pretrain decoder!!!!!!!!!!!!!!!!!!")
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

        #if self.is_train:
        #    # if self.mode == tf.estimator.ModeKeys.TRAIN:
        #    tf.logging.info("!!!!!! using rl predict in traning !!!!!!")
        #    decoded_ids, log_probs = self.rl_predict_new(self.encoder_outputs, self.attention_bias)
        #    return decoded_ids, log_probs
        #else:
        tf.logging.info("!!!!!!! using argmax_predict in prediction/evaluation !!!!!!!!")
        decoded_ids, sample_logprobs, sample_ids  = self.argmax_predict(self.encoder_outputs, self.attention_bias)
        return decoded_ids, sample_logprobs, sample_ids 

    #def build_rollout_generator(self, gen_targets, encoder_outputs, given_num, roll_steps=2):
    #    """
    #    :param gen_targets: [batch, src_len, enc_size]
    #    :param encoder_outputs: [batch, gen_tgt_len]
    #    :param given_num: [batch,1,1 src_len]
    #    :return:
    #    """
    #    gen_tgt_len = tf.shape(gen_targets)[1]
    #    roll_gen_tgt = gen_targets[:, :given_num + 1]

    #    def inner_loop(i, given_num, roll_gen_tgt):
    #        roll_logits = self.decode(roll_gen_tgt, encoder_outputs,
    #                                  self.attention_bias)  # [batch, given_num + 1, vocab_size]
    #        roll_one_step = tf.multinomial(roll_logits[:, given_num, :], num_samples=1,
    #                                       output_dtype=tf.int32)  # [batch, 1]
    #        roll_gen_tgt = tf.concat([roll_gen_tgt, roll_one_step], axis=1)  # [batch, tgt_len]
    #        return i + 1, given_num + 1, roll_gen_tgt

    #    _, given_num, roll_gen_tgt = tf.while_loop(
    #        cond=lambda i, given_num, _: (i < roll_steps) & (given_num < gen_tgt_len),
    #        body=inner_loop,
    #        loop_vars=[tf.constant(0), given_num, roll_gen_tgt],
    #        shape_invariants=[
    #            tf.TensorShape([]),
    #            given_num.get_shape(),
    #            tf.TensorShape([None, None])]
    #    )
    #    roll_gen_tgt = tf.concat([roll_gen_tgt, gen_targets[:, given_num + 1:]], axis=1)
    #    roll_logits = self.decode(roll_gen_tgt, encoder_outputs, self.attention_bias)
    #    roll_samples = tf.multinomial(tf.reshape(roll_logits, [-1, self.params.target_vocab_size]),
    #                                  num_samples=1, output_dtype=tf.int32)  # [batch * gen_tgt_len, vocab_size]
    #    batch_size = tf.shape(encoder_outputs)[0]
    #    roll_samples = tf.reshape(roll_samples, [batch_size, -1])
    #    decoder_ids = tf.concat([roll_gen_tgt[:, :given_num], roll_samples[:, given_num:]], axis=1)  # [batch, tgt_len]
    #    return decoder_ids

    #def build_rollout_generator(self, gen_targets, encoder_outputs, given_num, roll_steps=4):
    #    """
    #    :param gen_targets: [batch, src_len, enc_size]
    #    :param encoder_outputs: [batch, gen_tgt_len]
    #    :param given_num: [batch,1,1 src_len]
    #    :return:
    #    """
    #    gen_tgt_len = tf.shape(gen_targets)[1]
    #    roll_gen_tgt = gen_targets[:, :given_num + 1]

    #    def inner_loop(i, given_num, roll_gen_tgt):
    #        roll_logits = self.decode(roll_gen_tgt, encoder_outputs,
    #                                  self.attention_bias)  # [batch, given_num + 1, vocab_size]
    #        roll_one_step = tf.multinomial(roll_logits[:, given_num, :], num_samples=1,
    #                                       output_dtype=tf.int32)  # [batch, 1]
    #        roll_gen_tgt = tf.concat([roll_gen_tgt, roll_one_step], axis=1)  # [batch, tgt_len]
    #        return i + 1, given_num + 1, roll_gen_tgt

    #    _, given_num, roll_gen_tgt = tf.while_loop(
    #        cond=lambda i, given_num, _: (i < roll_steps) & (given_num < gen_tgt_len),
    #        body=inner_loop,
    #        loop_vars=[tf.constant(0), given_num, roll_gen_tgt],
    #        shape_invariants=[
    #            tf.TensorShape([]),
    #            given_num.get_shape(),
    #            tf.TensorShape([None, None])]
    #    )
    #    roll_gen_tgt = tf.concat([roll_gen_tgt, gen_targets[:, given_num + 1:]], axis=1)
    #    # roll_logits = self.decode(roll_gen_tgt, encoder_outputs, self.attention_bias)  # teacher forcing
    #    # roll_samples = tf.multinomial(tf.reshape(roll_logits, [-1, self.params.target_vocab_size]),
    #    #                               num_samples=1, output_dtype=tf.int32)  # [batch * gen_tgt_len, vocab_size]
    #    # batch_size = tf.shape(encoder_outputs)[0]
    #    # roll_samples = tf.reshape(roll_samples, [batch_size, -1])
    #    # decoder_ids = tf.concat([roll_gen_tgt[:, :given_num], roll_samples[:, given_num:]], axis=1)  # [batch, tgt_len]
    #    decoder_ids = roll_gen_tgt
    #    return decoder_ids


    def build_rollout_generator(self, ac_gen_tgt, log_probs, encoder_outputs, given_num, roll_steps, min_len):
        """
        :param gen_targets: [batch, src_len, enc_size]
        :param encoder_outputs: [batch, gen_tgt_len]
        :param given_num: [batch,1,1 src_len]
        :return:
        """
        #batch_size = tf.shape(gen_targets)[0]
        #gen_tgt_len = tf.shape(gen_targets)[1]
        #action = tf.reshape(gen_sample_ids[:, given_num], (-1,1))
        ##action = tf.multinomial(distribution, num_samples=1, output_dtype=tf.int32)
        #roll_gen_tgt = tf.concat([gen_targets[:, :given_num], action], axis=1)
        #log_probs = tf.reshape(sample_logprobs[:, given_num], (-1,1))
        #print("roll_gen_tgt", roll_gen_tgt)
        log_probs = tf.reshape(log_probs, (-1, 1))
        roll_gen_tgt = ac_gen_tgt
        
        def inner_loop(i, given_num, roll_gen_tgt, log_probs):
            roll_gen_tgt = tf.pad(roll_gen_tgt, [[0, 0], [0, 1]])  # [batch, given_num+1]
            print("padded roll_gen_tgt", roll_gen_tgt, roll_gen_tgt.shape)
            roll_logits = self.decode(roll_gen_tgt, encoder_outputs,
                                      self.attention_bias)  # [batch, given_num + 1, vocab_size]
            print("roll_logits", roll_logits.shape)
            categorical = tf.contrib.distributions.Categorical(logits=roll_logits[:, -1, :])
            roll_one_step = categorical.sample()            # [batch, 1]
            log_prob = categorical.log_prob(roll_one_step) # [batch,] 
            log_prob = tf.reshape(log_prob, (-1, 1))
            log_probs = tf.concat([log_probs, log_prob], axis=1)
            roll_one_step = tf.reshape(roll_one_step, (-1,1))
            roll_gen_tgt = tf.concat([roll_gen_tgt[:, :-1], roll_one_step], axis=1)  # [batch, tgt_len]
            return i + 1, given_num + 1, roll_gen_tgt, log_probs

        _, given_num, roll_gen_tgt, log_probs = tf.while_loop(
            cond=lambda i, given_num, _1, _2: (i < roll_steps) & (given_num < min_len),
            body=inner_loop,
            loop_vars=[tf.constant(0), given_num, roll_gen_tgt, log_probs],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None])]
        )
        #roll_gen_tgt = tf.concat([roll_gen_tgt, gen_targets[:, given_num + 1:]], axis=1)
        print("roll_gen_tgt,", roll_gen_tgt)
        #roll_logits = self.decode(roll_gen_tgt, encoder_outputs, self.attention_bias)
        #roll_samples = tf.multinomial(tf.reshape(roll_logits, [-1, self.params.target_vocab_size]),
        #                              num_samples=1, output_dtype=tf.int32)  # [batch * gen_tgt_len, vocab_size]
        #batch_size = tf.shape(encoder_outputs)[0]
        #roll_samples = tf.reshape(roll_samples, [batch_size, -1])
        #decoder_ids = tf.concat([roll_gen_tgt[:, :given_num], roll_samples[:, given_num:]], axis=1)  # [batch, tgt_len]
        #return decoder_ids
        return roll_gen_tgt, log_probs[:, 0]
    
    def build_padding_rollout_generator(gen_targets, enc_outputs, min_len):
        
        init_given_num = tf.constant(0, dtype=tf.int32)
        


    def build_bleu_discriminator(self, origin_inputs, gen_targets):
        fake_prediction = self.build_pretrain(gen_targets, targets=None)  # argmax_predict
        fake_bleu = tf.py_func(metrics.compute_bleu_batch, (origin_inputs, fake_prediction), tf.float32)
        # reward = metrics.compute_bleu(reference_corpus=origin_inputs,
        #                               translation_corpus=fake_prediction)
        tf.identity(fake_bleu[:5], "fake_bleu")
        cur_reward = fake_bleu
        return cur_reward 

    def get_reward_bleu_1(self, origin_inputs, gen_targets, gen_sample_ids, sample_logprobs, roll_num):
        baseline = self.get_fake_blue_teach(origin_inputs, gen_targets)
        tf.identity(baseline[:5], "baseline")
        
        total_loss = 0

        print("gen_targets,", gen_targets)
        lengths = tf.reduce_sum(tf.cast(tf.not_equal(gen_targets, 0), tf.int32), axis=1)
        min_len = lengths[tf.math.argmin(lengths)]
        given_num = tf.argmax(tf.concat([tf.ones((1,), dtype=tf.float32) * -1e10,
                                         tf.random_normal((min_len,), dtype=tf.float32)[1:]], axis=0),
                              output_type=tf.int32)
        print("given_num,", given_num)
        
        batch_size = tf.shape(gen_targets)[0]
        gen_tgt_len = tf.shape(gen_targets)[1]
        action = tf.reshape(gen_sample_ids[:, given_num - 1], (-1,1))
        ac_gen_tgt = tf.concat([gen_targets[:, :given_num - 1], action], axis=1) # [batch, given_num]
        print("ac_gen_tgt", ac_gen_tgt)
        log_probs = tf.reshape(sample_logprobs[:, given_num - 1], (-1,1))

        roll_steps = 3
        rewards = []
        for i in range(roll_num):
            print("roll_num:", i)
            roll_samples, log_porbs = self.build_rollout_generator(
                ac_gen_tgt, log_probs, self.encoder_outputs, given_num, roll_steps=roll_steps, min_len=min_len) 
            roll_samples = tf.concat([roll_samples, gen_targets[:, given_num + roll_steps:]], axis=1)
            print("roll_samples,", roll_samples)
            cur_reward = self.build_bleu_discriminator(
                origin_inputs=origin_inputs,
                gen_targets=roll_samples
            )
            rewards.append(tf.reshape(cur_reward, (-1, 1)))
        rewards = tf.concat(rewards, axis=1)    # [batch, roll_nums]
        tf.identity(rewards[:5], "origin_rewards")
        rewards = tf.maximum(rewards - tf.reshape(baseline, (-1, 1)), 0.0)
        #rewards /= tf.sqrt(tf.reduce_mean(rewards) + 1e-12)
        tf.identity(rewards[:5], "rewards")
        total_loss += log_probs * tf.stop_gradient(rewards)
        tf.identity(log_probs, "log_probs")
        return - tf.reduce_mean(total_loss) / roll_num

    #def hard_choice_given_num(self, gen_targets, origin_targets):
    #    """
    #    :param gen_targets:  [batch, gen_tgt_len]
    #    :param origin_targets: [batch, ori_tgt_len]
    #    :return:
    #    """
    #    def get_given_num(gen_targets, origin_targets):
    #        gen_targets_list = gen_targets.tolist()
    #        origin_targets_list = origin_targets.tolist()
    #        diffs = list(set(gen_targets_list[0]) - set(origin_targets_list[0]))
    #        token = diffs[0]
    #        given_num = 1
    #        for idx in range(len(gen_targets_list)):
    #            if gen_targets_list[idx] == token:
    #                given_num = idx
    #        return np.int32(given_num)
    #    given_num = tf.py_func(get_given_num, (gen_targets, origin_targets), tf.int32)
    #    given_num = tf.identity(given_num, "given_num")
    #    return given_num

    #def get_reward_bleu_2(self, origin_inputs, gen_targets, origin_target, roll_num, log_probs):
    #    total_loss = 0
    #    given_num = self.hard_choice_given_num(gen_targets, origin_target)
    #    print(given_num)
    #    for i in range(roll_num):
    #        roll_samples = self.build_rollout_generator(
    #            gen_targets, self.encoder_outputs, given_num)  # [batch, tgt_len]
    #        cur_reward = self.build_bleu_discriminator(
    #            origin_inputs=origin_inputs,
    #            gen_targets=roll_samples
    #        )
    #        total_loss += log_probs[:, given_num] * tf.stop_gradient(cur_reward)
    #    return - tf.reduce_mean(total_loss) / roll_num


    def get_real_blue_teach(self, origin_inputs, origin_targets):
        real_logits = self.build_pretrain(origin_targets, targets=origin_inputs)
        real_prediction = tf.argmax(real_logits, axis=-1) # [batch, ori_inp_len]
        comp_bleu_time = time.time()
        real_bleu = tf.py_func(metrics.compute_bleu_batch, (origin_inputs, real_prediction), tf.float32)
        tf.logging.info("comp_bleu_time: {}".format(time.time() - comp_bleu_time))
        return real_bleu

    def get_fake_blue_teach(self, origin_inputs, gen_targets):
        fake_logits = self.build_pretrain(gen_targets, targets=origin_inputs)  # argmax_predict
        fake_prediction = tf.argmax(fake_logits, axis=-1)  # [batch, ori_inp_len]
        fake_bleu = tf.py_func(metrics.compute_bleu_batch, (origin_inputs, fake_prediction), tf.float32)
        return fake_bleu

    #def get_reward_bleu_teach(self, origin_inputs, gen_targets, origin_targets, roll_num, log_probs):
    #    total_loss = 0
    #    #tgt_len = tf.shape(gen_targets)[1]
    #    real_bleu = self.get_real_blue_teach(origin_inputs, origin_targets=origin_targets)
    #    baseline = self.get_real_blue_teach(gen_targets, origin_targets=origin_targets) 
    #    lengths = tf.reduce_sum(tf.cast(tf.not_equal(gen_targets, 0), tf.int32), axis=1)
    #    min_len = lengths[tf.math.argmin(lengths)] 
    #    given_num = tf.argmax(tf.concat([tf.ones((1,), dtype=tf.float32) * -1e10,
    #                                     tf.random_normal((min_len,), dtype=tf.float32)[1:]], axis=0),
    #                          output_type=tf.int32)
    #    fake_bleu_total = 0
    #    for i in range(roll_num):
    #        roll_time = time.time()
    #        roll_samples = self.build_rollout_generator(
    #            gen_targets, self.encoder_outputs, given_num)  # [batch, tgt_len]
    #        tf.logging.info("roll_time: {}".format(time.time() - roll_time))
    #        fake_bleu = self.get_fake_blue_teach(
    #            origin_inputs=origin_inputs,
    #            gen_targets=roll_samples,
    #        )
    #        fake_bleu_total += fake_bleu
    #    tf.identity((fake_bleu_total/roll_num), "fake_bleu")
    #    tf.identity(real_bleu, "real_bleu")
    #    #reward = 1 / (tf.maximum(0.01, real_bleu - fake_bleu_total/roll_num) * 100)
    #    #reward = tf.maximum(0.01, (fake_bleu_total/roll_num) / (real_bleu + 1e-6))
    #    
    #    #reward = tf.minimum(tf.maximum(0.0, reward - tf.reduce_mean(reward)), 1.0)
    #    #reward = fake_bleu - baseline
    #    reward = tf.maximum(fake_bleu - tf.reduce_mean(fake_bleu), 0.0)
    #    tf.identity(reward, "reward")
    #    tf.identity(tf.reduce_mean(fake_bleu), "mean_reward")
    #    total_loss += log_probs[:, given_num] * tf.stop_gradient(reward)
    #    return - tf.reduce_mean(total_loss)


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
            #def inner_loop(i, finished, next_id, decoded_ids, decoded_logits, sample_logprobs, sample_ids, cache):
            def inner_loop(i, finished, next_id, decoded_ids, sample_logprobs, sample_ids, cache):
                """One step of greedy decoding."""
                logits, cache = symbols_to_logits_fn(next_id, i, cache)  # [batch, vocab_size]
                next_id = tf.argmax(logits, -1, output_type=tf.int32)
                categorical = tf.contrib.distributions.Categorical(logits=logits)
                sample_id = categorical.sample()
                sample_logprob = categorical.log_prob(sample_id)
                finished |= tf.equal(next_id, EOS_ID)

                next_id = tf.reshape(next_id, shape=[-1, 1])
                decoded_ids = tf.concat([decoded_ids, next_id], axis=1)

                #logits = tf.expand_dims(logits, axis=1)
                #decoded_logits = tf.concat([decoded_logits, logits], axis=1)


                sample_logprob = tf.reshape(sample_logprob, (-1,1))
                sample_logprobs = tf.concat([sample_logprobs, sample_logprob],axis=1)

                sample_id = tf.reshape(sample_id, (-1, 1))
                sample_ids = tf.concat([sample_ids, sample_id], axis=1)
                #return i + 1, finished, next_id, decoded_ids, decoded_logits, sample_logprobs, sample_ids, cache
                return i + 1, finished, next_id, decoded_ids, sample_logprobs, sample_ids, cache

            def is_not_finished(i, finished, _1, _2, _3, _4, _5):
                return (i < max_decode_length) & tf.logical_not(tf.reduce_all(finished))

            #decoded_logits = tf.zeros([batch_size, 0, self.params.target_vocab_size], dtype=tf.float32)
            sample_logprobs = tf.zeros([batch_size, 0], dtype=tf.float32)
            decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int32)
            sample_ids = tf.zeros([batch_size, 0], dtype=tf.int32)
            finished = tf.fill([batch_size], False)
            next_id = tf.zeros([batch_size, 1], dtype=tf.int32)

            #_, _, _, decoded_ids, decoded_logits, sample_logprobs, sample_ids, _ = tf.while_loop(
            _, _, _, decoded_ids, sample_logprobs, sample_ids, _ = tf.while_loop(
                cond=is_not_finished,
                body=inner_loop,
                loop_vars=[tf.constant(0), finished, next_id, decoded_ids, sample_logprobs, sample_ids, cache],
                shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    nest.map_structure(get_state_shape_invariants, cache),
                ])

            #return decoded_ids, decoded_logits, sample_logprobs, sample_ids
            return decoded_ids, sample_logprobs, sample_ids

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

            # return {"outputs": decoded_ids, "scores": tf.ones([batch_size, 1])}
            return decoded_ids, log_probs

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

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    tf.enable_eager_execution()
    x_inputs = tf.constant([[1, 2, 3, 0, 0], [3, 4, 5, 6, 0]], dtype=tf.int32)
    y_target = tf.constant([[5, 6, 7, 8, 20], [7, 3, 2, 6, 5]], dtype=tf.int32)
    params = model_params.TransformerBaseParams()
    model = Transformer(params, ModeKeys.TRAIN == "train")

    # test generator
    gen_targets, sample_logprobs, sample_ids = model.build_generator(x_inputs)
    #g_loss = model.get_reward_bleu_1(x_inputs, gen_targets, gen_sample_ids=sample_ids, sample_logprobs=sample_logprobs, roll_num=3)
    #g_loss_2 = model.get_reward_bleu_1(x_inputs, gen_targets, origin_target=y_target, gen_sample_ids=sample_ids, sample_logprobs=sample_logprobs, roll_num=3)
    #print(g_loss)
    print(sample_logprobs)
    print(sample_ids)

