import tensorflow as tf
from model import model_params
from config import flags_obj
import os
from utils import dataset, metrics, tokenizer
from six.moves import xrange  # pylint: disable=redefined-builtin
from model import transformer_8
import re, time, math
import numpy as np
from datetime import datetime

TOWER_NAME = "tower"
MOVING_AVERAGE_DECAY = 0.9999

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

fp = open(os.path.join(flags_obj.data_dir, 'vocab.bpe.' + str(flags_obj.vocabulary) + "." + flags_obj.fro), 'r')
lines = fp.readlines()
params.source_vocab_size = len(lines)
fp = open(os.path.join(flags_obj.data_dir, 'vocab.bpe.' + str(flags_obj.vocabulary) + "." + flags_obj.to), 'r')
lines = fp.readlines()
params.target_vocab_size = len(lines)

vocab_file_source = os.path.join(flags_obj.data_dir,
                                 'vocab' + '.bpe.' + str(flags_obj.search) + '.' + flags_obj.fro)
vocab_file_target = os.path.join(flags_obj.data_dir, 'vocab' + '.bpe.' + str(flags_obj.search) + '.' + flags_obj.to)
subtokenizer_source = tokenizer.Subtokenizer.init_from_files(vocab_file_source, flags_obj.search)
subtokenizer_target = tokenizer.Subtokenizer.init_from_files(vocab_file_target, flags_obj.search)

if params.shared_embedding_softmax_weights:
    assert params.target_vocab_size == params.source_vocab_size
    params.vocab_size = params.source_vocab_size
    tf.logging.info("sharing vocab size:{}".format(params.vocab_size))
else:
    tf.logging.info("source vocab size:{}, target vocab size:{}".format
                    (params.source_vocab_size, params.target_vocab_size))


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


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def gan_tower_loss(scope, g_model, d_model, input_fn):
    """ calculate the total loss on a single tower runing the train model.

    :param scope:
    :param src:
    :param tgt:
    :return:
    """
    # Build inference Graph.
    gen_samples = g_model.build_pretrain(input_fn.source, targets=None)
    g_loss, fake_loss = d_model.gan_loss(input_fn.source, input_fn.target, gen_samples, 0)
    
    tf.add_to_collection("g_losses", g_loss)
    g_losses = tf.get_collection("g_losses", scope)
    total_g_loss = tf.add_n(g_losses, name="total_g_loss")
    
    tf.add_to_collection("fake_losses", fake_loss)
    fake_losses = tf.get_collection("fake_losses", scope)
    total_fake_loss = tf.add_n(fake_losses, name="total_fake_loss")
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in fake_losses + [total_fake_loss] + g_losses + [total_g_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)
    return total_fake_loss, total_g_loss


def evaluation(model, input_fn):
    tf.logging.info("!!!Build graph for evaluation!!!")
    logits = model.build_pretrain(input_fn.source, input_fn.target)
    xentropy, weights = metrics.padded_cross_entropy_loss(
        logits, input_fn.target, params.label_smoothing, params.target_vocab_size)
    loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
    return loss, logits, input_fn.target

def update_checkpoint(pretrain_dir, model_dir, replace_from, replace_to):
    pre_checkpoint = tf.train.get_checkpoint_state(pretrain_dir)
    for var_name, _ in tf.contrib.framework.list_variables(pretrain_dir):
        print(var_name)
    print("!!!numbers of variables", len(tf.contrib.framework.list_variables(pretrain_dir)))
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(pretrain_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(pretrain_dir, var_name)

            # Set the new name
            new_name = var_name
            new_name = new_name.replace(replace_from, replace_to)
            if replace_from in var_name or "beta" in var_name or "step" in var_name:
                print('Updating %s to %s.' % (var_name, new_name))
                var_dis = tf.Variable(var, name=new_name)
                var = tf.Variable(var, name=var_name)

        print(len(tf.trainable_variables()))
        #for var in tf.trainable_variables():
        #    print(var)
        # Save the variables
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        save_checkpoint_path = os.path.join(flags_obj.model_dir, 'model.ckpt')
        saver.save(sess, save_checkpoint_path, global_step=0)
        print("Update variable name from {}, and saving in {}".format(pre_checkpoint.model_checkpoint_path, save_checkpoint_path + "-" + str(0)))

def array_to_string(samples):
    string = ""
    for ids in samples:
        token = subtokenizer_target.subtoken_list[ids]
        string = string + token + " "
    return string


def train(params):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # calculate the learning rate schedule
        learning_rate = get_learning_rate(
            params.learning_rate, params.hidden_size,
            params.learning_rate_warmup_steps,
            global_step)

        optimizer = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate,
            beta1=params.optimizer_adam_beta1,
            beta2=params.optimizer_adam_beta2,
            epsilon=params.optimizer_adam_epsilon)

        # get src,tgt sentence for each model tower
        my_dataset = dataset.Dataset(params)
        train_iterator = my_dataset.train_input_fn(params)
        valid_iterator = my_dataset.eval_input_fn(params)
        
        f_tower_grads = []
        g_tower_grads = []
        g_model = transformer_8.Transformer(params, is_train=True)
        d_model = transformer_8.Discriminator(params, is_train=True)

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            for i in xrange(flags_obj.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                        tf.logging.info("Build graph on gpu:{}".format(i))
                        #fake_loss, g_loss = gan_tower_loss(scope, g_model, d_model, train_iterator)
                        tf.logging.info("Generate samples")
                        gen_samples = g_model.build_pretrain(input_fn.source, targets=None)
                        
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        f_grads = optimizer.compute_gradients(fake_loss)
                        g_grads = optimizer.compute_gradients(g_loss)
                        dis_g_grads = []
                        dis_f_grads = []
                        total_size = 0

                        for grad, var in g_grads:
                            if "Discriminator" in var.name:
                                v_size = np.prod(np.array(var.shape.as_list())).tolist()
                                total_size += v_size
                                dis_g_grads.append((grad, var))
                        for grad, var in f_grads:
                            if "Discriminator" in var.name:
                                dis_f_grads.append((grad, var))
                        tf.logging.info("total trainable variables number: {}".format(len(dis_g_grads)))
                        tf.logging.info("Total trainable variables size: %d", total_size)
                        g_tower_grads.append(dis_g_grads)
                        f_tower_grads.append(dis_f_grads)
                    if i == 0 and valid_iterator:
                        val_loss_op, val_logits_op, val_tgt_op = evaluation(g_model, valid_iterator)
                        # summaries.append(tf.summary.scalar("val_loss", val_loss_op))

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        if len(g_tower_grads) > 1:
            g_grads = average_gradients(g_tower_grads)
            f_grads = average_gradients(f_tower_grads)
        else:
            g_grads = g_tower_grads[0]
            f_grads = f_tower_grads[0]

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', learning_rate))

        # Add histograms for gradients.
        for grad, var in g_grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        g_apply_gradient_op = optimizer.apply_gradients(g_grads, global_step=global_step)
        f_apply_gradient_op = optimizer.apply_gradients(f_grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #    MOVING_AVERAGE_DECAY, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(g_apply_gradient_op, f_apply_gradient_op)
        # train_op = g_apply_gradient_op

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True

        with tf.Session(config=sess_config) as sess:
            sess.run(init)
            sess.run(tf.local_variables_initializer())

            sess.run(train_iterator.initializer)
            
            #ckpt = tf.train.latest_checkpoint(flags_obj.pretrain_dir)
            ckpt = tf.train.latest_checkpoint(flags_obj.model_dir)
            tf.logging.info("ckpt {}".format(ckpt))
            if ckpt and tf.train.checkpoint_exists(ckpt):
                tf.logging.info("Reloading model parameters..from {}".format(ckpt))
                saver.restore(sess, ckpt)
            else:
                tf.logging.info("Create a new model...{}".format(flags_obj.model_dir))

            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)
            summary_writer = tf.summary.FileWriter(flags_obj.model_dir, sess.graph)

            best_bleu = 0.0
            for step in xrange(flags_obj.train_steps):
                start_time = time.time()
                _, g_loss_value, r_loss, f_loss = sess.run([train_op, g_loss, d_model.real_loss, d_model.fake_loss])
                tf.logging.info("step = {}, step_g_loss = {:.4f}, r_loss = {:.4f}, f_loss = {:.4f}".
                                format(step, g_loss_value, r_loss, f_loss))
                duration = time.time() - start_time

                assert not np.isnan(g_loss_value), 'Model diverged with loss = NaN'

                if step % 100 == 0:
                    num_examples_per_step = flags_obj.batch_size * flags_obj.num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / flags_obj.num_gpus

                    format_str = ('%s: step %d, g_loss = %.4f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    tf.logging.info(format_str % (datetime.now(), step, g_loss_value,
                                                  examples_per_sec, sec_per_batch))


                if step % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                if step % flags_obj.steps_between_evals == 0:
                    sess.run(valid_iterator.initializer)
                    tf.logging.info(
                        "-------------------- Validation step ...{} -------------------------- ----------".format(step))
                    total_bleu = 0.0
                    total_size = 0
                    total_loss = 0.0
                    while True:
                        try:
                            val_loss, val_logit, val_tgt = sess.run([val_loss_op, val_logits_op, val_tgt_op])
                            val_pred = np.argmax(val_logit, axis=-1)
                            val_bleu = metrics.compute_bleu(val_tgt, val_pred)
                            batch_size = val_pred.shape[0]
                            total_bleu += val_bleu * batch_size
                            total_loss += val_loss * batch_size
                            total_size += batch_size
                            tf.logging.info("pairs shape {}, {}, step_bleu: {:.5f}, step_loss: {:.4f}".
                                            format(val_pred.shape, val_tgt.shape, val_bleu, val_loss))
                        except tf.errors.OutOfRangeError:
                            pred_string = array_to_string(val_pred[-1])
                            tgt_string = array_to_string(val_tgt[-1])
                            tf.logging.info("prediction:\n{}".format(pred_string))
                            tf.logging.info("target:\n{}".format(tgt_string))
                            tf.logging.info("Finished going through the valid dataset")
                            break
                    total_bleu /= total_size
                    total_loss /= total_size
                    tf.logging.info(
                        "{}, Step: {}, Valid loss: {:.6f}, Valid bleu : {:.6f}".format(datetime.now(), step, total_loss,
                                                                                       total_bleu))
                    tf.logging.info(
                        "--------------------- Finish evaluation -----------------------------------------------------")
                    # Save the model checkpoint periodically.
                    if step == 0:
                        total_bleu = 0.0

                    if total_bleu > best_bleu:
                        best_bleu = total_bleu
                        checkpoint_path = os.path.join(flags_obj.model_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
                        tf.logging.info("Saving model at {}".format(checkpoint_path + "-" + str(step)))


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(flags_obj.model_dir):
        # tf.gfile.DeleteRecursively(flags_obj.model_dir)
        # tf.logging.info("flags_obj.model_dir")
        pass
    else:
        tf.gfile.MakeDirs(flags_obj.model_dir)
    train(params)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    update_checkpoint(flags_obj.pretrain_dir, flags_obj.model_dir, "Transformer", "Discriminator")
    tf.app.run()


