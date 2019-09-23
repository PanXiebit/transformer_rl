from model.transformer_9 import Transformer
import tensorflow as tf
from model.base import ModeKeys
from model import model_utils
from utils import metrics
from utils.tokenizer import EOS_ID, PAD_ID

class Generator(Transformer):
    def __init__(self, params, is_train, name_scope, mode=None):
        super(Generator, self).__init__(params, is_train, mode=mode, scope=name_scope)
        self.name_scope = name_scope

    def build_padding_rollout_generator(self, real_inputs, gen_samples, max_len, given_num):
        with tf.variable_scope(self.name_scope, initializer=self.initializer, reuse=tf.AUTO_REUSE):
            if ModeKeys.is_predict_one(self.mode):
                self.attention_bias = None
            else:
                self.attention_bias = model_utils.get_padding_bias(real_inputs)
            self.encoder_outputs = self.encode(real_inputs, self.attention_bias)

            def condition(given_num, _):
                return given_num < max_len

            def inner_loop(given_num, given_y):
                logits = self.decode(given_y, self.encoder_outputs, self.attention_bias)
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

    def get_reward(self, real_inputs, real_targets, gen_targets, roll_num, discriminator):
        real_loss = discriminator.get_loss(real_targets, real_inputs)  # [batch ,1]
        max_len = tf.shape(gen_targets)[1]
        lengths = tf.reduce_sum(tf.cast(tf.not_equal(gen_targets, PAD_ID), tf.int32), axis=1)
        min_len = lengths[tf.argmin(lengths)]

        given_num = tf.argmax(
            tf.concat([tf.ones((1,), dtype=tf.float32) * (-1e5),
                       tf.random_normal([min_len - 1, ], dtype=tf.float32)],
                      axis=0), output_type=tf.int32)

        total_rewards = []
        for i in range(roll_num):
            tf.logging.info("roll_num: {}".format(i))
            roll_sample = self.build_padding_rollout_generator(
                real_inputs=real_inputs,
                gen_samples=gen_targets,
                max_len=max_len,
                given_num=given_num)
            roll_loss = discriminator.get_loss(
                gen_targets=roll_sample,
                origin_inputs=real_inputs)  # [batch ,1]
            cur_reward = 1 / tf.maximum(roll_loss / real_loss, 1)
            total_rewards.append(cur_reward)  # list, [batch,1] * roll_num
        total_rewards = tf.reduce_mean(tf.concat(total_rewards, axis=1), axis=1)  # [bacth, roll_num] -> [batch, ]
        return given_num, total_rewards

    def g_loss(self, gen_targets, given_num, rewards):
        given_y = gen_targets[:, :given_num]
        with tf.variable_scope(self.name_scope, initializer=self.initializer, reuse=tf.AUTO_REUSE):
            logits = self.decode(targets=given_y, encoder_outputs=self.encoder_outputs,
                                 attention_bias=self.attention_bias)
            batch_size = tf.shape(gen_targets)[0]
            probs = tf.nn.softmax(logits[:, -1, :], axis=-1)  # probability, [batch, dec_vocab_size]
            log_probs = tf.reduce_sum(tf.one_hot(given_y[:, -1], self.params.target_vocab_size, 1.0, 0.0) * tf.log(
                tf.clip_by_value(probs, 1e-20, 1.0)), axis=1)
            rewards = tf.stop_gradient(rewards)
            g_loss = - tf.reduce_sum(log_probs * tf.reshape(rewards, [-1])) / tf.to_float(batch_size)
            return g_loss


class Discriminator(Transformer):
    def __init__(self, params, is_train, name_scope, mode=None):
        super(Discriminator, self).__init__(params, is_train, mode=None, scope=name_scope)
        self.name_scope = name_scope

    def get_loss(self, gen_targets, real_inputs):
        with tf.variable_scope(self.name_scope, initializer=self.initializer, reuse=tf.AUTO_REUSE):
            logits = self.inference(gen_targets, real_inputs)
            xentropy, weights = metrics.padded_cross_entropy_loss(logits, real_inputs,
                                                                  self.params.label_smoothing,
                                                                  self.params.target_vocab_size)
            loss = tf.reduce_sum(xentropy, axis=1) / tf.reduce_sum(weights, axis=1)  # [batch, 1]
            return tf.reshape(loss, (-1, 1))
