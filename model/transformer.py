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
import numpy as np

EOS_ID = 1
_NEG_INF = -1e9


# params = model_params.TransformerBaseParams()

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

            # softmax层的线性变换的矩阵与embedding共享，还有这种操作。。
            # 但是仔细一想，很合理啊，得到的vector与embdding中的向量之间做内积，然后取argmax.
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

        # Calculate attention bias for encoder self-attention and decoder
        # multi-headed attention layers.
        if ModeKeys.is_predict_one(self.mode):
            attention_bias = None
        else:
            attention_bias = model_utils.get_padding_bias(inputs)  # [batch, 1, 1, src_len]

        # Run the inputs through the encoder layer to map the symbol
        # representations to continuous representations.
        encoder_outputs = self.encode(inputs, attention_bias)  # [batch, src_len, hidden_size]

        # get encdec_attenion k/v just for predict_one_encoder
        if self.mode == ModeKeys.PREDICT_ONE_ENCODER:
            fake_decoder_inputs = tf.zeros([1, 0, self.params.hidden_size])
            fake_decoder_outputs = self.decoder_stack(fake_decoder_inputs, encoder_outputs, None, None, None)

        # Generate output sequence if targets is None, or return logits if target
        # sequence is known.
        if targets is None:
            tf.logging.info("!!!!!!!!!!!prediction using argmax prediction!!!!!!!!!!!!!")
            prediction, _ = self.argmax_predict(encoder_outputs, attention_bias)
            return prediction
        else:
            logits = self.decode(targets, encoder_outputs, attention_bias)   # [batch, tgt_len, vocab_size]
            return logits

    def build_generator(self, inputs):
        # Calculate attention bias for encoder self-attention and decoder
        # multi-headed attention layers.
        if ModeKeys.is_predict_one(self.mode):
            self.attention_bias = None
        else:
            self.attention_bias = model_utils.get_padding_bias(inputs)  # [batch, 1, 1, src_len]

        # Run the inputs through the encoder layer to map the symbol
        # representations to continuous representations.
        self.encoder_outputs = self.encode(inputs, self.attention_bias)  # [batch, src_len, hidden_size]

        # get encdec_attenion k/v just for predict_one_encoder
        if self.mode == ModeKeys.PREDICT_ONE_ENCODER:
            fake_decoder_inputs = tf.zeros([1, 0, self.params.hidden_size])
            fake_decoder_outputs = self.decoder_stack(fake_decoder_inputs, self.encoder_outputs, None, None, None)

        # Generate output sequence if targets is None, or return logits if target
        # sequence is known.
        if self.is_train:
            tf.logging.info("!!!!!! using rl predict in traning !!!!!!")
            return self.rl_predict(self.encoder_outputs, self.attention_bias)
        else:
            tf.logging.info("!!!!!!! using argmax_predict in inference !!!!!!!!")
            return self.argmax_predict(self.encoder_outputs, self.attention_bias)


    def build_teacher_force_rollout_generator(self, gen_targets, encoder_outputs, given_num):
        """
        :param encoder_outputs:  [batch, src_len, enc_size]
        :param gen_targets:      [batch, gen_tgt_len]
        :param encoder_decoder_attention_bias:  [batch,1,1 src_len]
        :return:
        """
        roll_logits = self.decode(gen_targets, encoder_outputs, self.attention_bias)  # [batch, gen_tgt_len, vocab_size]
        roll_samples = tf.multinomial(tf.reshape(roll_logits, [-1, self.params.target_vocab_size]),
                                      num_samples=1, output_dtype=tf.int32)  # [batch * gen_tgt_len, vocab_size]
        batch_size = tf.shape(encoder_outputs)[0]
        roll_samples = tf.reshape(roll_samples, [batch_size, -1])
        decoder_ids = tf.concat([gen_targets[:, :given_num], roll_samples[:, given_num:]], axis=1)  # [batch, tgt_len]
        print("decoder_ids", decoder_ids.shape)
        return decoder_ids

    def build_new_rollout_generator(self, gen_targets, encoder_outputs, given_num, roll_steps=4):
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

    def build_rollout_generator(self, encoder_outputs, gen_targets, encoder_decoder_attention_bias, given_num):
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
            def inner_loop_1(i, finished, next_id, decoded_ids, cache):
                logits, cache = symbols_to_logits_fn(next_id, i, cache)
                next_id = tf.reshape(gen_targets[:, i], (batch_size, 1))
                finished |= tf.equal(next_id, EOS_ID)
                finished = tf.reshape(finished, (-1,))
                decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
                return i + 1, finished, next_id, decoded_ids, cache

            def inner_loop_2(i, finished, next_id, decoded_ids, cache):
                """One step of greedy decoding."""
                logits, cache = symbols_to_logits_fn(next_id, i, cache)
                # next_id = tf.argmax(logits, -1, output_type=tf.int32)
                next_id = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
                finished |= tf.equal(next_id, EOS_ID)
                finished = tf.reshape(finished, (-1,))
                # next_id = tf.expand_dims(next_id, axis=1)
                next_id = tf.reshape(next_id, shape=[-1, 1])
                decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
                return i + 1, finished, next_id, decoded_ids, cache

            def is_not_finished_1(i, finished, _1, _2, _3):
                return (i < given_num) & tf.logical_not(tf.reduce_all(finished))

            def is_not_finished_2(i, finished, _1, _2, _3):
                return (i < max_decode_length) & tf.logical_not(tf.reduce_all(finished))

            decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int32)
            finished = tf.fill([batch_size], False)
            next_id = tf.zeros([batch_size, 1], dtype=tf.int32)

            # i < given number
            i, finished, next_id, decoded_ids, cache = tf.while_loop(
                cond=is_not_finished_1,
                body=inner_loop_1,
                loop_vars=[tf.constant(0), finished, next_id, decoded_ids, cache],
                shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    nest.map_structure(get_state_shape_invariants, cache),
                ])
            # print("seen time step: ", i)
            _, _, _, decoded_ids, _ = tf.while_loop(
                cond=is_not_finished_2,
                body=inner_loop_2,
                loop_vars=[i, finished, next_id, decoded_ids, cache],
                shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    nest.map_structure(get_state_shape_invariants, cache),
                ])
            return decoded_ids

    def get_real_loss(self, origin_inputs, origin_target):
        real_logits = self.build_pretrain(inputs=origin_target, targets=origin_inputs)  # [batch, tgt_len, vocab_size]
        real_xentropy, real_weights = metrics.padded_cross_entropy_loss(
            real_logits, origin_inputs, self.params.label_smoothing, self.params.target_vocab_size)
        real_loss = tf.reduce_sum(real_xentropy, axis=1) / tf.reduce_sum(real_weights, axis=1) # [batch]
        tf.identity(real_loss[:5], "real_loss")
        mean_real_loss = tf.reduce_mean(real_loss, name="mean_real_loss")
        tf.summary.scalar("mean_real_loss", mean_real_loss)
        return real_loss

    def build_discriminator(self, origin_inputs, gen_target, margin, real_loss, given_num=None, discount_factor=0.95):
        fake_logits = self.build_pretrain(inputs=gen_target, targets=origin_inputs)  # [batch, tgt_length, vocab_size]
        fake_xentropy, fake_weights = metrics.padded_cross_entropy_loss(
            fake_logits, origin_inputs, self.params.label_smoothing, self.params.target_vocab_size) # [batch, origin_length]
        #print("fake_xentropy:", fake_xentropy.shape)
        #print("-------given_num-----", given_num) 
        #fake_xentropy = tf.transpose(fake_xentropy, perm=[1, 0])  # [tgt_len, batch]
        #tgt_len = tf.shape(fake_xentropy)[0] 
        #
        #def _unstack_ta(inp):
        #    return tf.TensorArray(
        #        dtype=inp.dtype, size=tf.shape(inp)[0],
        #        element_shape=inp.get_shape()[1:]).unstack(inp)
        #
        #ta_fake_xentropy = nest.map_structure(_unstack_ta, fake_xentropy)

        #def _create_ta(inp):
        #    return tf.TensorArray(
        #        dtype=tf.float32,
        #        size=tgt_len,
        #        dynamic_size=False,
        #        element_shape=inp.get_shape()[1:])

        #discounted_fake_loss = nest.map_structure(_create_ta, fake_xentropy)

        #def inner_loop_1(i, ta_fake_xentropy, discounted_fake_loss):
        #    print("aaaaa", (i, given_num))
        #    disc_loss = ta_fake_xentropy.read(i)
        #    discounted_fake_loss = nest.map_structure(lambda ta, out: ta.write(i, out),
        #                                              discounted_fake_loss, disc_loss)
        #    return i + 1, ta_fake_xentropy, discounted_fake_loss
        #
        #def inner_loop_2(i, ta_fake_xentropy, discounted_fake_loss):
        #    print("bbbbbb", (i, tgt_len))
        #    disc_loss = ta_fake_xentropy.read(i) * (discount_factor ** tf.to_float(i - given_num))
        #    discounted_fake_loss = nest.map_structure(lambda ta, out: ta.write(i, out),
        #                                              discounted_fake_loss, disc_loss)
        #    return i + 1, ta_fake_xentropy, discounted_fake_loss
        #
        ## i < given_num
        #i, ta_fake_xentropy, discounted_fake_loss = tf.while_loop(
        #    cond=lambda i, _1, _2: tf.less(i, given_num),
        #    body=inner_loop_1,
        #    loop_vars=[tf.constant(0), ta_fake_xentropy, discounted_fake_loss],
        #)
        ## # # i >= given_num
        #i, ta_fake_xentropy, discounted_fake_loss = tf.while_loop(
        #    cond=lambda i, _1, _2: tf.less(i, tgt_len),
        #    body=inner_loop_2,
        #    loop_vars=[i, ta_fake_xentropy, discounted_fake_loss],
        #)

        #fake_loss = tf.transpose(discounted_fake_loss.stack(), perm=[1, 0])  # [batch, tgt_length]
        #fake_loss = tf.reduce_sum(fake_loss, axis=1) / tf.reduce_sum(fake_weights, axis=1)
        fake_loss = tf.reduce_sum(fake_xentropy, axis=1) / tf.reduce_sum(fake_weights, axis=1)

        tf.identity(fake_loss[:5], "fake_loss")
        mean_fake_loss = tf.reduce_mean(fake_loss, name="mean_fake_loss")
        tf.summary.scalar("mean_fake_loss", mean_fake_loss)
        rewards = 1 / tf.maximum(0.2, fake_loss/(real_loss + 1e-12) - 1)  # [batch]
        tf.identity(rewards[:5], "rewards")
        mean_wards = tf.reduce_mean(rewards, name="mean_wards")
        tf.summary.scalar("mean_wards", mean_wards)
        return rewards

    #def get_reward(self, origin_inputs, gen_targets, origin_target, roll_num, margin):
    #    # Calculate attention bias for encoder self-attention and decoder
    #    # multi-headed attention layers.
    #    # if ModeKeys.is_predict_one(self.mode):
    #    #     attention_bias = None
    #    # else:
    #    #     attention_bias = model_utils.get_padding_bias(origin_inputs)  # [batch, 1, 1, src_len]
    #    # encoder_outputs = self.encode(origin_inputs, attention_bias)  # [batch, src_len, hidden_size]
    #    # Run the origin_inputs through the encoder layer to map the symbol
    #    # representations to continuous representations.
    #    real_loss = self.get_real_loss(origin_target, origin_inputs)  # [batch]

    #    # get encdec_attenion k/v just for predict_one_encoder
    #    if self.mode == ModeKeys.PREDICT_ONE_ENCODER:
    #        fake_decoder_origin_inputs = tf.zeros([1, 0, self.params.hidden_size])
    #        fake_decoder_outputs = self.decoder_stack(fake_decoder_origin_inputs, self.encoder_outputs, None, None, None)

    #    total_loss = []
    #    for i in range(roll_num):
    #        tf.logging.info("roll_num: {}".format(i))
    #        roll_loss = tf.reduce_sum(tf.zeros_like(gen_targets, dtype=tf.float32), axis=1, keep_dims=True)  # [batch ,1]

    #        def condition(given_num, _):
    #            return tf.less(given_num, tf.shape(gen_targets)[1])

    #        def inner_loop(given_num, roll_loss):
    #            tf.logging.info("given_num: {}".format(given_num))
    #            roll_samples = self.build_rollout_generator(self.encoder_outputs, gen_targets, self.attention_bias, given_num)
    #            print("roll sample: {}".format(roll_samples.shape))
    #            cur_loss = self.build_discriminator(
    #                gen_target=roll_samples,
    #                origin_inputs=origin_inputs,
    #                real_loss=real_loss,
    #                margin=margin
    #            )
    #            # ypred = np.array([item[1] for item in cur_loss])
    #            roll_loss = tf.concat([roll_loss, tf.reshape(cur_loss, (-1, 1))], axis=1)
    #            given_num += 1
    #            return given_num, roll_loss

    #        _, roll_loss = tf.while_loop(
    #            cond=condition,
    #            body=inner_loop,
    #            loop_vars=[tf.constant(1), roll_loss],
    #            shape_invariants=[tf.TensorShape([]),
    #                              tf.TensorShape([None, None])]
    #        )

    #        # the last token reward, the whole sentence
    #        cur_loss = self.build_discriminator(
    #            gen_target=gen_targets,
    #            origin_inputs=origin_inputs,
    #            real_loss=real_loss,
    #            margin=margin
    #        )
    #        roll_loss = tf.concat([roll_loss, tf.reshape(cur_loss, (-1, 1))], axis=1)  # [batch, tgt_len+1]
    #        total_loss.append(tf.expand_dims(roll_loss, axis=-1))
    #    total_loss = tf.concat(total_loss, axis=-1)[:, 1:, :]   # []
    #    # total_loss = (total_loss - tf.reduce_mean(total_loss, axis=1, keepdims=True))   # [f_target_len, roll_num]
    #    total_loss = tf.reduce_sum(total_loss, axis=-1) / roll_num      # [batch, gen_tgt_len]
    #    return total_loss

    #def get_reward_one_timestep(self, origin_inputs, gen_targets, origin_target, roll_num, margin, log_probs=None):
    #    real_loss = self.get_real_loss(origin_target, origin_inputs)  # [batch]
    #    total_loss = 0
    #    for i in range(roll_num):
    #        given_num = tf.argmax(tf.random_normal(tf.shape(gen_targets), dtype=tf.float32)[0], output_type=tf.int32)
    #        roll_samples = self.build_rollout_generator(self.encoder_outputs, gen_targets, self.attention_bias,
    #                                                    given_num)
    #        cur_loss = self.build_discriminator(
    #        gen_target=roll_samples,
    #        origin_inputs=origin_inputs,
    #        real_loss=real_loss,
    #        margin=margin)
    #        tf.identity(given_num, "given_num")
    #        total_loss += log_probs[:, given_num] * tf.stop_gradient(cur_loss)
    #        tf.identity(log_probs[:, given_num], "log_probs")
    #    return tf.reduce_sum(total_loss) / roll_num

    #def gan_output(self, gen_targets, reward):
    #    gan_logits = self.decode(gen_targets, self.encoder_outputs, self.attention_bias)  # [batch, gen_tgt_len, vocab_size]
    #    with tf.variable_scope("output", initializer=self._initializer, reuse=tf.AUTO_REUSE):
    #        l_shape = tf.shape(gan_logits)
    #        probs = tf.nn.softmax(tf.reshape(gan_logits, [-1, self.params.target_vocab_size]))  #[batch * gen_tgt_len, vocab_size]
    #        sample = tf.to_float(l_shape[0])
    #        g_loss = tf.reduce_sum(tf.one_hot(tf.reshape(gen_targets, [-1]), self.params.target_vocab_size, 1.0, 0.0) * probs,
    #                          axis=1) * tf.reshape(reward, [-1])       # [batch * gen_tgt_len]
    #        g_loss = tf.reduce_sum(g_loss) / sample
    #    return g_loss


    def get_reward_teacher_force(self, origin_inputs, gen_targets, origin_target, roll_num, margin, log_probs=None):
        real_loss = self.get_real_loss(origin_target, origin_inputs)  # [batch]
        print("gen_target", gen_targets.shape)
        total_loss = 0
        tgt_len = tf.shape(gen_targets)[1]
        given_num = tf.argmax(tf.concat([tf.ones((1,), dtype=tf.float32) * -1e10,
                                         tf.random_normal((tgt_len,), dtype=tf.float32)[1:]], axis=0),
                              output_type=tf.int32) 
        #given_num = tf.argmax(tf.random_normal(tf.shape(gen_targets), dtype=tf.float32)[0], output_type=tf.int32)
        #if tf.math.equal(given_num, 0):
        #    print("!!!!!!!!!!! given_num is zero !!!!!!!!!!!!!!!!!!")
        #    given_num = tf.add(given_num, 1)

        for i in range(roll_num):
            #roll_samples = self.build_teacher_force_rollout_generator(
            #    gen_targets, self.encoder_outputs, given_num)          # [batch, tgt_len]
            roll_samples = self.build_new_rollout_generator(
                gen_targets, self.encoder_outputs, given_num)          # [batch, tgt_len]
            cur_reward = self.build_discriminator(
                gen_target=roll_samples,
                origin_inputs=origin_inputs,
                real_loss=real_loss,
                margin=margin,
                given_num=given_num)
            tf.identity(given_num, "given_num")
            total_loss += log_probs[:, given_num] * tf.stop_gradient(cur_reward)
            tf.identity(log_probs[:, given_num][:5], "log_probs")
        #return tf.reduce_mean(total_loss) / roll_num
        return  - tf.reduce_mean(total_loss) / roll_num, tf.reduce_mean(real_loss)

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
            def inner_loop(i, finished, next_id, decoded_ids, cache):
                print("time step:", i)
                """One step of greedy decoding."""
                logits, cache = symbols_to_logits_fn(next_id, i, cache)
                # logits, cache = symbols_to_logits_fn(decoded_ids, i, cache)
                next_id = tf.argmax(logits, -1, output_type=tf.int32)
                finished |= tf.equal(next_id, EOS_ID)
                # next_id = tf.expand_dims(next_id, axis=1)
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

            return decoded_ids, _

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
            def inner_loop(i, finished, next_id, decoded_ids, log_probs, cache):
                # print("time step:", i)
                """One step of greedy decoding."""

                logits, cache = symbols_to_logits_fn(next_id, i, cache)
                # next_id = tf.argmax(logits, -1, output_type=tf.int32)
                categorical = tf.contrib.distributions.Categorical(logits=logits)
                # next_id = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
                next_id = categorical.sample()
                log_prob = categorical.log_prob(next_id)  # [batch,]
                finished |= tf.equal(next_id, EOS_ID)   
                finished = tf.reshape(finished, (-1,))
                next_id = tf.reshape(next_id, shape=[-1, 1])
                log_prob = tf.reshape(log_prob, shape=[-1, 1])
                decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
                log_probs = tf.concat([log_probs, log_prob], axis=1)    # [batch, len]
                return i + 1, finished, next_id, decoded_ids, log_probs, cache

            def is_not_finished(i, finished, _1, _2, _3, _4):
                return (i < max_decode_length) & tf.logical_not(tf.reduce_all(finished))

            decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int32)
            log_probs =  tf.zeros([batch_size, 0], dtype=tf.float32)
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.enable_eager_execution()
    x_inputs = tf.constant([[1, 2, 3, 0, 0], [3, 4, 5, 6, 0]], dtype=tf.int32)
    y_target = tf.constant([[1, 2, 3, 0, 0], [3, 4, 5, 6, 0]], dtype=tf.int32)
    params = model_params.TransformerBaseParams()
    model = Transformer(params, is_train=True, mode=ModeKeys.TRAIN)

    # test pretrain
    pretrain_out = model.build_pretrain(x_inputs, y_target)
    print(pretrain_out.shape)
    
    # test generator
    gen_target, log_probs = model.build_generator(x_inputs)
    print(gen_target)
    roll_samples = model.build_new_rollout_generator(gen_target, model.encoder_outputs, given_num=3)
    print(roll_samples)
