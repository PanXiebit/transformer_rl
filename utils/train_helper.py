import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow
from utils.tokenizer import EOS_ID, PAD_ID

def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")

def rename(sess, pretrain_dir, model_dir, replace_from, replace_to, global_step):
    pre_checkpoint = tf.train.get_checkpoint_state(pretrain_dir)
    tf.logging.info(
        "Numbers of variables in pretrain_dir {}".format(len(tf.contrib.framework.list_variables(pretrain_dir))))
    var_to_save = []
    for var_name, _ in tf.contrib.framework.list_variables(pretrain_dir):
        var = tf.contrib.framework.load_variable(pretrain_dir, var_name)
        new_name = var_name
        new_name = new_name.replace(replace_from, replace_to)
        if replace_from in var_name or "beta" in var_name or "step" in var_name:
            if "Adam" not in var_name:
                var_dis = tf.Variable(var, name=new_name)
                var = tf.Variable(var, name=var_name)
                var_to_save.append(var)
                var_to_save.append(var_dis)

    tf.logging.info("The number of variables to save: {}".format(len(var_to_save)))
    sess.run(tf.initialize_variables(var_to_save))
    saver = tf.train.Saver(var_to_save)
    save_checkpoint_path = os.path.join(model_dir, 'model.ckpt')
    saver.save(sess, save_checkpoint_path, global_step=global_step)
    tf.logging.info("Update variable name from {}, and saving in {}".
                    format(pre_checkpoint.model_checkpoint_path, save_checkpoint_path + "-" + str(global_step)))


def update_checkpoint_2(var_list, model_dir, replace_from, replace_to):
    tf.logging.info("Numbers of variables in pretrain_dir {}".
                    format(len(tf.contrib.framework.list_variables(model_dir))))
    tf.logging.info("Loading parameters from {}".format(model_dir))
    ckpt_map = {}
    for var_name, _ in tf.contrib.framework.list_variables(model_dir):
        var_np = tf.contrib.framework.load_variable(model_dir, var_name)
        ckpt_map[var_name] = var_np

    update_op = []
    for var in var_list:
        var_name = var.name.split(":")[0]
        if replace_to in var_name:
            update_op.append(tf.assign(var, ckpt_map[var_name]))
        elif replace_from in var_name:
            new_name = var_name.replace(replace_from, replace_to)
            update_op.append(tf.assign(var, ckpt_map[new_name]))
        else:
            print(var)
    return update_op


def update_checkpoint(var_list, replace_from, replace_to):
    var_map = {}
    for var in var_list:
        var_map[var.name] = var

    update_op = []
    for var in var_list:
        var_name = var.name
        if replace_from in var_name:
            pass
        elif replace_to in var_name:
            if "Adam" not in var_name:
                new_name = var_name.replace(replace_to, replace_from)
                update_op.append(tf.assign(var, var_map[new_name]))
        else:
            print(var)
    return update_op


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def get_len(ids):
    try:
        index = list(ids).index(EOS_ID) + 1  # encoder need add EOS
        return index
    except ValueError:
        return len(list(ids))


def _trim_and_pad(targets):
    batch_size = tf.shape(targets)[0]
    init_lens = tf.zeros([0], dtype=tf.int64)

    def len_inner_loop(i, lengths):
        cur_tgt = targets[i]
        len = tf.py_func(get_len, [cur_tgt], tf.int64)
        len = tf.reshape(len, [1])
        lengths = tf.concat([lengths, len], axis=0)
        i += 1
        return i, lengths

    _, lengths = tf.while_loop(
        cond=lambda i, _: i < batch_size,
        body=len_inner_loop,
        loop_vars=[tf.constant(0), init_lens],
        shape_invariants=[tf.TensorShape([]), tf.TensorShape([None])]
    )

    max_len = lengths[tf.argmax(lengths)]

    pad_targets = tf.zeros([0, max_len], dtype=tf.int32)

    def inner_loop(i, pad_inputs):
        ori_length = lengths[i]
        ori_input = tf.reshape(targets[i][:ori_length], [1, -1])
        pad_input = tf.pad(ori_input, [[0, 0], [0, max_len - ori_length]])
        pad_inputs = tf.concat([pad_inputs, pad_input], axis=0)
        return i + 1, pad_inputs

    _, pad_targets = tf.while_loop(
        cond=lambda i, _: i < batch_size,
        body=inner_loop,
        loop_vars=[tf.constant(0), pad_targets],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None, None])]
    )

    return pad_targets

if __name__ == "__main__":
    import os
    from model import model_params, transformer_9
    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # tf.enable_eager_execution()
    x_inputs = tf.constant([[1, 2, 3, 0, 0], [3, 4, 5, 6, 0]], dtype=tf.int32)
    y_target = tf.constant([[5, 6, 7, 8, 20], [7, 3, 2, 6, 5]], dtype=tf.int32)
    params = model_params.TransformerBaseParams()


    ckpt1 = tf.train.latest_checkpoint()
    ckpt2 = tf.train.latest_checkpoint()
    len1 = len(tf.contrib.framework.list_variables(ckpt1))
    len2 = len(tf.contrib.framework.list_variables(ckpt2))
    print(len1, len2)

    var_list = []
    for var_name, _ in tf.contrib.framework.list_variables(ckpt1):
        var_list.append(var_name)
    for var_name, _ in tf.contrib.framework.list_variables(ckpt2):
        if var_name not in var_list:
            print(var_name)