import tensorflow as tf
from model import model_params
from config import flags_obj
import os
from utils import dataset, metrics, tokenizer
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
from datetime import datetime
import gen_and_dis
from utils import train_helper

TOWER_NAME = "tower"
MOVING_AVERAGE_DECAY = 0.9999

vocab_file_source = os.path.join(flags_obj.data_dir,
                                 'vocab' + '.bpe.' + str(flags_obj.search) + '.' + flags_obj.fro)
vocab_file_target = os.path.join(flags_obj.data_dir,
                                 'vocab' + '.bpe.' + str(flags_obj.search) + '.' + flags_obj.to)
subtokenizer_source = tokenizer.Subtokenizer.init_from_files(
    vocab_file_source, flags_obj.search)
subtokenizer_target = tokenizer.Subtokenizer.init_from_files(
    vocab_file_target, flags_obj.search)

def overwrite_parameters():
    PARAMS_MAP = {
        "base": model_params.TransformerBaseParams,
        "small": model_params.TransformerSmallParams,
    }

    params = PARAMS_MAP[flags_obj.param_set]
    params.data_dir = flags_obj.data_dir
    params.model_dir = flags_obj.model_dir
    params.pretrain_dir = flags_obj.pretrain_dir
    params.num_parallel_calls = flags_obj.num_parallel_calls
    params.batch_size = flags_obj.batch_size or params.batch_size
    params.learning_rate = flags_obj.learning_rate or params.learning_rate
    params.max_length = flags_obj.max_length or params.max_length
    params.is_reversed = flags_obj.is_reversed
    params.keep_checkpoint_max = flags_obj.keep_checkpoint_max
    params.save_checkpoints_secs = flags_obj.save_checkpoints_secs
    params.hvd = flags_obj.hvd
    params.repeat_dataset = -1
    params.shared_embedding_softmax_weights = flags_obj.shared_embedding_softmax_weights

    fp = open(os.path.join(flags_obj.data_dir, 'vocab.bpe.' +
                           str(flags_obj.vocabulary) + "." + flags_obj.fro), 'r')
    lines = fp.readlines()
    params.source_vocab_size = len(lines)
    fp = open(os.path.join(flags_obj.data_dir, 'vocab.bpe.' +
                           str(flags_obj.vocabulary) + "." + flags_obj.to), 'r')
    lines = fp.readlines()
    params.target_vocab_size = len(lines)

    if params.shared_embedding_softmax_weights:
        assert params.target_vocab_size == params.source_vocab_size
        params.vocab_size = params.source_vocab_size
        tf.logging.info("sharing vocab size:{}".format(params.vocab_size))
    else:
        tf.logging.info("source vocab size:{}, target vocab size:{}".format
                        (params.source_vocab_size, params.target_vocab_size))
    return params


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps, global_step):
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(global_step)
        learning_rate *= (hidden_size ** -0.5)
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))
        tf.identity(learning_rate, "learning_rate")
        tf.summary.scalar("learning_rate", learning_rate)
        return learning_rate


def array_to_string(samples):
    string = ""
    for ids in samples:
        token = subtokenizer_target.subtoken_list[ids]
        string = string + token + " "
    return string


def build_graph(params):
    my_dataset = dataset.Dataset(params)
    train_iterator = my_dataset.train_input_fn(params)
    valid_iterator = my_dataset.eval_input_fn(params)

    ckpt = tf.train.latest_checkpoint(flags_obj.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt):
        init_step = int(tf.train.latest_checkpoint(flags_obj.model_dir).split("-")[-1])
        global_step = tf.get_variable(
            'global_step',
            initializer=init_step,
            trainable=False)
    else:
        init_step = 0
        global_step = tf.Variable(init_step, trainable=False, name="global_step")

    learning_rate = get_learning_rate(
        params.learning_rate, params.hidden_size,
        params.learning_rate_warmup_steps,
        global_step)

    optimizer = tf.contrib.opt.LazyAdamOptimizer(
        learning_rate,
        beta1=params.optimizer_adam_beta1,
        beta2=params.optimizer_adam_beta2,
        epsilon=params.optimizer_adam_epsilon)

    tower_grads = []
    g_tower_grads = []
    g_model = gen_and_dis.Generator(params, is_train=True, name_scope="Transformer")
    d_model = gen_and_dis.Discriminator(params, is_train=True, name_scope="Discriminator")
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for i in xrange(flags_obj.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                    tf.logging.info("Build graph on gpu:{}".format(i))

                    # pretrain loss
                    logits = g_model.inference(train_iterator.source, train_iterator.target)
                    xentropy, weights = metrics.padded_cross_entropy_loss(
                        logits, train_iterator.target, params.label_smoothing, params.target_vocab_size)
                    xen_loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

                    # g_loss
                    gen_samples = g_model.inference(train_iterator.source, None)
                    deal_samples = train_helper._trim_and_pad(gen_samples)

                    given_num, rewards = g_model.get_reward(real_inputs=train_iterator.source,
                                                            real_targets=deal_samples,
                                                            gen_targets=train_iterator.target,
                                                            roll_num=flags_obj.roll_num,
                                                            discriminator=d_model)
                    g_loss = g_model.g_loss(gen_targets=deal_samples,
                                            given_num=given_num,
                                            rewards=rewards)

                    grads = optimizer.compute_gradients(xen_loss)
                    g_grads = optimizer.compute_gradients(g_loss)
                    tf.logging.info("total trainable variables number: {}".format(len(grads)))
                    tower_grads.append(grads)
                    g_tower_grads.append(g_grads)

                if i == 0 and valid_iterator:
                    val_pred = g_model.inference(inputs=valid_iterator.source,
                                                      targets=None)

    if len(tower_grads) > 1:
        grads = train_helper.average_gradients(tower_grads)
        g_grads = train_helper.average_gradients(g_tower_grads)
    else:
        grads = tower_grads[0]
        g_grads = g_tower_grads[0]

    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    g_apply_gradient_op = optimizer.apply_gradients(g_grads, global_step=global_step)

    train_op = tf.group(apply_gradient_op, g_apply_gradient_op)

    train_return = (train_op, global_step, g_loss, xen_loss, rewards, learning_rate, init_step)
    valid_return = (val_pred, valid_iterator.target, valid_iterator.source)
    dataset_iter = (train_iterator, valid_iterator)
    return g_model, d_model, train_return, valid_return, dataset_iter

def train(params):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        g_model, d_model, train_return, valid_return, dataset_iter = build_graph(params)
        train_op, global_step, g_loss, xen_loss, rewards, learning_rate, init_step = train_return
        val_pred, val_tgt, val_src = valid_return
        train_iterator, valid_iterator = dataset_iter


        vars_to_update = tf.global_variables()
        update_op = train_helper.update_checkpoint(vars_to_update, replace_from="Transformer",
                                                   replace_to="Discriminator")

        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=20)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True



        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            sess.run(train_iterator.initializer)

            # reload the parameters
            ckpt = tf.train.latest_checkpoint(flags_obj.model_dir)
            tf.logging.info("ckpt {}".format(ckpt))
            if ckpt and tf.train.checkpoint_exists(ckpt):
                tf.logging.info("Reloading model parameters..from {}".format(ckpt))
                variables = tf.global_variables()
                var_keep_dic = train_helper.get_variables_in_checkpoint_file(ckpt)
                var_keep_dic.pop('global_step')

                variables_to_restore = []
                for v in variables:
                    if v.name.split(':')[0] in var_keep_dic:
                        variables_to_restore.append(v)
                restorer = tf.train.Saver(variables_to_restore)
                restorer.restore(sess, ckpt)
            else:
                tf.logging.info("Create a new model...{}".format(flags_obj.model_dir))

            tf.train.start_queue_runners(sess=sess)
            summary_writer = tf.summary.FileWriter(flags_obj.model_dir, sess.graph)

            best_bleu = 0.0
            for step in xrange(init_step, flags_obj.train_steps):

                # Train generator for 5 steps
                tf.logging.info("Training generator")
                g_steps_per_iter = 5
                for g_step in range(g_steps_per_iter):
                    _, x_loss_value, g_loss_value, rewards_value = sess.run(
                        [train_op, xen_loss, g_loss, rewards])

                    assert not np.isnan(g_loss_value), 'Model diverged with loss = NaN'
                    assert not np.isnan(x_loss_value), 'Model diverged with loss = NaN'

                    if step % 100 == 0:
                        tf.logging.info(
                            "step = {}, g_loss = {:.4f}, x_loss = {:.4f}, reward = {}".
                            format(step, g_loss_value, x_loss_value, rewards_value[:5]))

                # train discriminator
                sess.run(update_op)

                if step % flags_obj.steps_between_evals == 0:
                    sess.run(valid_iterator.initializer)
                    tf.logging.info(
                        "------------- Validation step ...{} -----------".format(step))
                    total_bleu = 0.0
                    total_size = 0
                    while True:
                        try:
                            val_tgt_np, val_src_np, val_pred_np = sess.run([val_tgt, val_src, val_pred])
                            val_bleu = metrics.compute_bleu(val_tgt_np, val_pred_np)
                            batch_size = val_pred_np.shape[0]
                            total_bleu += val_bleu * batch_size
                            total_size += batch_size
                        except tf.errors.OutOfRangeError:
                            break
                    total_bleu /= total_size
                    tf.logging.info(
                        "{}, Step: {}, Valid bleu : {:.6f}".format(
                            datetime.now(), step, total_bleu))

                    tf.logging.info(
                        "--------------------- Finish evaluation ---------------------")
                    # Save the model checkpoint periodically.
                    if total_bleu > best_bleu:
                        best_bleu = total_bleu
                        checkpoint_path = os.path.join(flags_obj.model_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
                        tf.logging.info("Saving model at {}".format(checkpoint_path + "-" + str(step)))


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(flags_obj.model_dir):
        tf.gfile.DeleteRecursively(flags_obj.model_dir)
        tf.logging.info("flags_obj.model_dir")
        pass
    else:
        tf.gfile.MakeDirs(flags_obj.model_dir)
    params = overwrite_parameters()
    train(params)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
