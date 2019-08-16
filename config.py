# from absl import flags
# from absl import app
import os
import tensorflow as tf
import multiprocessing

flags = tf.app.flags

# def define_transformer_flags():
flags.DEFINE_string(
    name="data_dir", short_name="dd", default="/home/work/xiepan/xp_dial/gan_nmt/transformer_rl/data/en-tr/v0/gen_data",
    help="The location of the input data.")

flags.DEFINE_string(
    name="data_dir_mono", short_name="mdd", default="/home/work/xiepan/xp_dial/gan_nmt/transformer_rl/data/en-tr/v0/gen_data",
    help="The location of the input data.")

flags.DEFINE_string(
    name="model_dir", short_name="md", default="/home/work/xiepan/xp_dial/gan_nmt/transformer_rl/model_save",
    help="The location of the model checkpoint files.")

flags.DEFINE_string(
    name="pretrain_dir", short_name="pmd", default="/home/work/xiepan/xp_dial/gan_nmt/transformer_rl/model_save",
    help="The location of the model checkpoint files.")

flags.DEFINE_integer(
    name="train_epochs", short_name="te", default=None,
    help="The number of epochs used to train.")

flags.DEFINE_integer(
    name="roll_num", short_name="rn", default=1,
    help="The number of epochs used to train.")

flags.DEFINE_integer(
    name="epochs_between_evals", short_name="ebe", default=1,
    help="The number of training epochs to run between "
         "evaluations.")

flags.DEFINE_float(
    name="stop_threshold", short_name="st",
    default=None,
    help="If passed, training will stop at the earlier of "
         "train_epochs and when the evaluation metric is  "
         "greater than or equal to stop_threshold.")

flags.DEFINE_float(
    name="learning_rate", short_name="lr",
    default=None,
    help="learning rate.")

flags.DEFINE_float(
    name="learning_rate_bt", short_name="lr_bt",
    default=None,
    help="learning rate back translation.")

flags.DEFINE_integer(
    name="batch_size", short_name="bs", default=10,
    help="Batch size for training and evaluation.")

flags.DEFINE_integer(
    name="max_length", short_name="ml", default=5,
    help="Maximum length of example.")
# assert not (multi_gpu and num_gpu)

flags.DEFINE_bool(
    name="multi_gpu", default=True,
    help="If set, run across all available GPUs.")

flags.DEFINE_integer(
    name="num_gpus", short_name="ng",
    # default=1 if tf.test.is_gpu_available() else 0,
    default=1,
    help="How many GPUs to use with the DistributionStrategies API. The "
         "default is 1 if TensorFlow can detect a GPU, and 0 otherwise.")

# Add transformer-specific flags
flags.DEFINE_string(
    name="param_set", short_name="mp", default="base",
    # enum_values=["base", "big", "beam_mid_1shard_4", "beam_mid_1shard_4_d0", "beam_mid_1shard_test4"],
    help="Parameter set to use when creating and training the model. The "
         "parameters define the input shape (batch size and max length), "
         "model configuration (size of embedding, # of hidden layers, etc.), "
         "and various other settings. The big parameter set increases the "
         "default batch size, embedding/hidden size, and filter size. For a "
         "complete list of parameters, please see model/model_params.py.")
flags.DEFINE_bool(
    name="shared_embedding_softmax_weights", default="false",
    help="Whether to use hvd")

flags.DEFINE_bool(
    name="hvd", short_name="hvd", default="false",
    help="Whether to use hvd")

# Flags for training with steps (may be used for debugging)
flags.DEFINE_integer(
    name="train_steps", short_name="ts", default=None,
    help="The number of steps used to train.")

flags.DEFINE_bool(
    name="is_reversed", short_name="rev", default=False,
    help=
    "Whether to reverse the dataset.")

flags.DEFINE_integer(
    name="steps_between_evals", short_name="sbe", default=1000,
    help=
    "The Number of training steps to run between evaluations. This is "
    "used if --train_steps is defined.")

# BLEU score computation
flags.DEFINE_string(
    name="bleu_source", short_name="bls", default=None,
    help=
    "Path to source file containing text translate when calculating the "
    "official BLEU score. --bleu_source, --bleu_ref, and --vocab_file "
    "must be set. Use the flag --stop_threshold to stop the script based "
    "on the uncased BLEU score.")
flags.DEFINE_string(
    name="bleu_ref", short_name="blr", default=None,
    help=
    "Path to source file containing text translate when calculating the "
    "official BLEU score. --bleu_source, --bleu_ref, and --vocab_file "
    "must be set. Use the flag --stop_threshold to stop the script based "
    "on the uncased BLEU score.")
flags.DEFINE_string(
    name="problem", default=None,
    help="problem.")
flags.DEFINE_integer(
    name="search", default=30000,
    help="Must set,if we use our own datas, use binary search to find the vocabulary set with size"
         "closest to the target size .")
flags.DEFINE_string(
    name="fro", default="src",
    help="problem.")
flags.DEFINE_string(
    name="to", default="tgt",
    help="problem.")

flags.DEFINE_string(
    name="decode_path", default="decode.txt",
    help="path to save decode result.")
flags.DEFINE_integer(
    name="keep_checkpoint_max", default=30,
    help=".")
flags.DEFINE_integer(
    name="save_checkpoints_secs", default=1200,
    help="")

flags.DEFINE_integer(
    name="vocabulary", default=30000,
    help=
    "Name of vocabulary file containing subtokens for subtokenizing the "
    "bleu_source file. This file is expected to be in the directory "
    "defined by --data_dir.")
flags.DEFINE_integer(
    name="beam_size", default=1,
    help=
    "the size of beam search")

flags.DEFINE_integer(
    name="num_parallel_calls", short_name="npc",
    default=multiprocessing.cpu_count(),
    help="The number of records that are  processed in parallel "
         "during input processing. This can be optimized per "
         "data set but for generally homogeneous data sets, "
         "should be approximately the number of available CPU "
         "cores. (default behavior)")

### debug parameters
flags.DEFINE_bool(
    name="debug", default=True,
    help="Use debugger to track down bad values during training.")

flags.DEFINE_string(
    name="ui_type", default="curses",
    help="Command-line user interface type (curses | readline)")

flags.DEFINE_string(
    name="dump_root", default="",
    help="Optional custom root directory for temporary debug dump data")

@flags.multi_flags_validator(
    ["train_epochs", "train_steps"],
    message="Both --train_steps and --train_epochs were set. Only one may be "
            "defined.")
def _check_train_limits(flag_dict):
    return flag_dict["train_epochs"] is None or flag_dict["train_steps"] is None


def _check_bleu_files(flags_dict):
    """Validate files when bleu_source and bleu_ref are defined."""
    if flags_dict["bleu_source"] is None or flags_dict["bleu_ref"] is None:
        return True
    # Ensure that bleu_source, bleu_ref, and vocab files exist.
    vocab_file_path_source = os.path.join(
        flags_dict["data_dir"], "vocab_" + str(flags_dict["vocabulary"]) + "." + flags_dict['fro'])
    vocab_file_path_target = os.path.join(
        flags_dict["data_dir"], "vocab_" + str(flags_dict["vocabulary"]) + "." + flags_dict['to'])

    return all([
        tf.gfile.Exists(flags_dict["bleu_source"]),
        tf.gfile.Exists(flags_dict["bleu_ref"]),
        tf.gfile.Exists(vocab_file_path_source),
        tf.gfile.Exists(vocab_file_path_target)])


flags_obj = flags.FLAGS

if __name__ == "__main__":
    # define_transformer_flags()
    print(flags_obj.model_dir)
