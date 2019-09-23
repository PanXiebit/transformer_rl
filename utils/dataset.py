# -*- encoding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from utils.tokenizer import PAD_ID
import collections


_FILE_SHUFFLE_BUFFER = 100  # 用来shuffle的文件数量
_READ_RECORD_BUFFER = 8 * 1000 * 1000

# Example grouping constants. Defines length boundaries for each group.
# These values are the defaults used in Tensor2Tensor.
_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1



class Dataset(object):
    def __init__(self, params):
        self.params = params


    def _read_and_batch_from_tfrecord(self, file_pattern, batch_size,
                                      max_length, num_parallel_calls, shuffle,
                                      repeat, is_reversed = False):
        dataset = tf.data.Dataset.list_files(file_pattern)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

        # ----------------------1. 读取tfrecord 文件,解析为 string tensor, ---------------
        # ------------------------- tf.data.TFRecordDataset ---------------------
        # 得到 serialized_example, A scalar string Tensor, a single serialized Example.
        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                self._load_records, sloppy=shuffle, cycle_length=num_parallel_calls))

        # -----------------------2. 逐行解析 string Tensor，转换成 int tensor -----------------
        # -----------------------  tf.parse_single_example ---------------------------------
        if not is_reversed:
            dataset = dataset.map(self._parse_example, num_parallel_calls=num_parallel_calls)

        # ------------------------3. 删除掉长度超过 maximum length 的example -------------------
        # ----------------------------- dataset.filter -------------------------------------
        dataset = dataset.filter(lambda x, y: self._filter_max_length((x, y), max_length))

        # -----------------------4. 按照长度相似的example进行group，并返回batched dataset --------------------
        dataset = self._batch_examples(dataset, batch_size, max_length)

        # -----------------------5. 重复多少个epoch -----------------------------
        dataset = dataset.repeat(repeat)

        # Prefetch the next element to improve speed of input pipeline.
        dataset = dataset.prefetch(-1)
        return dataset

    # ----------------------1. 读取tfrecord 文件,解析为 string tensor, ---------------
    def _load_records(self, filename):
        """Read file and return a dataset of tf.Examples."""
        return tf.data.TFRecordDataset(filename, buffer_size=_READ_RECORD_BUFFER)

    # -----------------------2. 逐行解析 string Tensor，转换成 int tensor -----------------
    def _parse_example(self, serialized_example):
      """Return inputs and targets Tensors from a serialized tf.Example."""
      data_fields = {
          "inputs": tf.VarLenFeature(tf.int64),
          "targets": tf.VarLenFeature(tf.int64)
      }
      parsed = tf.parse_single_example(serialized_example, data_fields)
      inputs = tf.sparse_tensor_to_dense(parsed["inputs"])
      targets = tf.sparse_tensor_to_dense(parsed["targets"])
      return inputs, targets

    # ------------------------3. 删除掉长度超过 maximum length 的example -------------------
    def _filter_max_length(self, example, max_length=50):
        """Indicates whether the example's length is lower than the maximum length."""
        # tf.size 计算所有的元素个数，这里一个example对应的size就是长度
        return tf.logical_and(tf.size(example[0]) <= max_length,
                              tf.size(example[1]) <= max_length)

    # -----------------------4. 按照长度相似的example进行group，并返回batched dataset --------------------
    def _batch_examples(self, dataset, batch_size, max_length):
        buckets_min, buckets_max = self._create_min_max_boundaries(max_length)
        # [0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 33, 36, 39, 42, 46]
        # [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 33, 36, 39, 42, 46, 51]
        # print(buckets_min, buckets_max)

        bucket_batch_sizes = [batch_size // x for x in buckets_max]
        bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
        print("bucket_batch_sizes: ", bucket_batch_sizes)
        def example_to_bucket_id(example_input, example_target):
            """Return int64 bucket id for this example, calculated based on length."""
            seq_length = self._get_example_length((example_input, example_target))

            # TODO: investigate whether removing code branching improves performance.
            conditions_c = tf.logical_and(
                tf.less_equal(buckets_min, seq_length), # Tensor("LessEqual_1:0", shape=(24,), dtype=bool)
                tf.less(seq_length, buckets_max))
            bucket_id = tf.reduce_min(tf.where(conditions_c))
            return bucket_id  # Tensor("Min:0", shape=(), dtype=int64)

        def window_size_fn(bucket_id):
            """Return number of examples to be grouped when given a bucket id."""
            return bucket_batch_sizes[bucket_id]

        def batching_fn(bucket_id, grouped_dataset):
            """Batch and add padding to a dataset of elements with similar lengths."""
            bucket_batch_size = window_size_fn(bucket_id)

            # Batch the dataset and add padding so that all input sequences in the
            # examples have the same length, and all target sequences have the same
            # lengths as well. Resulting lengths of inputs and targets can differ.
            return grouped_dataset.padded_batch(bucket_batch_size, ([None], [None]))

        return dataset.apply(tf.contrib.data.group_by_window(
            key_func=example_to_bucket_id,
            reduce_func=batching_fn,
            window_size=None,
            window_size_func=window_size_fn))


    # 定义bucket的最小和最大边界，以及
    def _create_min_max_boundaries(
            self, max_length, min_boundary=_MIN_BOUNDARY, boundary_scale=_BOUNDARY_SCALE):
        """Create min and max boundary lists up to max_length.

        For example, when max_length=24, min_boundary=4 and boundary_scale=2, the
        returned values will be:
          buckets_min = [0, 4, 8, 16, 24]
          buckets_max = [4, 8, 16, 24, 25]

        Args:
          max_length: The maximum length of example in dataset.
          min_boundary: Minimum length in boundary.
          boundary_scale: Amount to scale consecutive boundaries in the list.

        Returns:
          min and max boundary lists

        """
        # Create bucket boundaries list by scaling the previous boundary or adding 1
        # (to ensure increasing boundary sizes).
        bucket_boundaries = []
        x = min_boundary
        while x < max_length:
            bucket_boundaries.append(x)
            x = max(x + 1, int(x * boundary_scale))

        # Create min and max boundary lists from the initial list.
        buckets_min = [0] + bucket_boundaries
        buckets_max = bucket_boundaries + [max_length + 1]
        return buckets_min, buckets_max

    def _get_example_length(self, example):
        """Returns the maximum length between the example inputs and targets."""
        length = tf.maximum(tf.shape(example[0])[0], tf.shape(example[1])[0])
        return length

    def train_input_fn(self, params):
        # 文件名模板
        file_pattern = os.path.join(getattr(params, "data_dir", ""), "*train*")
        print("file_pattern", file_pattern)
        is_reversed = getattr(params, 'is_reversed')
        dataset = self._read_and_batch_from_tfrecord(file_pattern, params.batch_size,
                                                     params.max_length,
                                                     params.num_parallel_calls,
                                                     shuffle=True,
                                                     repeat=params.repeat_dataset,
                                                     is_reversed=is_reversed)
        # iterator = dataset.make_one_shot_iterator()
        # src, tgt = iterator.get_next()
        # return src, tgt
        iterator = dataset.make_initializable_iterator()
        src, tgt = iterator.get_next()
        return BatchedInput(initializer=iterator.initializer,
                            source=src,
                            target=tgt)

    def train_input_fn_mono(self, params):
        # 文件名模板
        file_pattern = os.path.join(getattr(params, "data_dir_mono", ""), "*train*")
        print("file_pattern", file_pattern)
        is_reversed = getattr(params, 'is_reversed')
        dataset = self._read_and_batch_from_tfrecord(file_pattern, params.batch_size,
                                                     params.max_length,
                                                     params.num_parallel_calls,
                                                     shuffle=True,
                                                     repeat=params.repeat_dataset,
                                                     is_reversed=is_reversed)
        #return dataset
        # iterator = dataset.make_one_shot_iterator()
        # src, tgt = iterator.get_next()
        # return src, tgt
        iterator = dataset.make_initializable_iterator()
        src, tgt = iterator.get_next()
        return BatchedInput(initializer=iterator.initializer,
                            source=src,
                            target=tgt)

    def eval_input_fn(self, params):
        """Load and return dataset of batched examples for use during evaluation."""
        file_pattern = os.path.join(getattr(params, "data_dir", ""), "*valid*")
        is_reversed = getattr(params, 'is_reversed')

        dataset = self._read_and_batch_from_tfrecord(
            file_pattern, params.batch_size, params.max_length,
            params.num_parallel_calls, shuffle=False, repeat=1, is_reversed=is_reversed)
        #return dataset
        # iterator = dataset.make_one_shot_iterator()
        # src, tgt = iterator.get_next()
        # return src, tgt
        iterator = dataset.make_initializable_iterator()
        src, tgt = iterator.get_next()
        return BatchedInput(initializer=iterator.initializer,
                            source=src,
                            target=tgt)


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target"))):
    pass

class MyDataset(object):
    def __init__(self, params):
        self.params = params

    def _read_and_batch_from_tfrecord(self, file_pattern, batch_size,
                                      max_length, num_parallel_calls, shuffle,
                                      repeat, is_reversed = False):
        dataset = tf.data.Dataset.list_files(file_pattern)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                self._load_records, sloppy=shuffle, cycle_length=num_parallel_calls)) 

        if not is_reversed:
            dataset = dataset.map(self._parse_example, num_parallel_calls=num_parallel_calls)

        dataset = dataset.filter(lambda x, y: self._filter_max_length((x, y), max_length))
        
        pad_value = tf.cast(tf.constant(PAD_ID), tf.int32)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            #padded_shapes=(tf.Dimension(max_length), tf.Dimension(max_length)),
            padded_shapes=(
                [tf.Dimension(max_length)],  # src
                [tf.Dimension(max_length)]),
            drop_remainder=True)
            #padding_values=(pad_value, 
            #    pad_value))
        dataset = dataset.repeat(repeat)

        # Prefetch the next element to improve speed of input pipeline.
        dataset = dataset.prefetch(-1)
        return dataset

    def _load_records(self, filename):
        """Read file and return a dataset of tf.Examples."""
        return tf.data.TFRecordDataset(filename, buffer_size=_READ_RECORD_BUFFER)
    
    def _filter_max_length(self, example, max_length=50):
        """Indicates whether the example's length is lower than the maximum length."""
        # tf.size 计算所有的元素个数，这里一个example对应的size就是长度
        return tf.logical_and(tf.size(example[0]) <= max_length,
                              tf.size(example[1]) <= max_length)
    
    def _parse_example(self, serialized_example):
        """Return inputs and targets Tensors from a serialized tf.Example."""
        data_fields = {
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.VarLenFeature(tf.int64)
        }
        parsed = tf.parse_single_example(serialized_example, data_fields)
        inputs = tf.sparse_tensor_to_dense(parsed["inputs"])
        targets = tf.sparse_tensor_to_dense(parsed["targets"])
        return inputs, targets
    
    def train_input_fn(self, params):
        # 文件名模板
        file_pattern = os.path.join(getattr(params, "data_dir", ""), "*train*")
        is_reversed = getattr(params, 'is_reversed')
        dataset = self._read_and_batch_from_tfrecord(file_pattern, params.batch_size,
                                                     params.max_length,
                                                     params.num_parallel_calls,
                                                     shuffle=True,
                                                     repeat=params.repeat_dataset,
                                                     is_reversed=is_reversed)
        #return dataset
        # iterator = dataset.make_one_shot_iterator()
        iterator = dataset.make_initializable_iterator()
        src, tgt = iterator.get_next()
        return BatchedInput(initializer=iterator.initializer,
                            source=src,
                            target=tgt)

    def eval_input_fn(self, params):
        """Load and return dataset of batched examples for use during evaluation."""
        file_pattern = os.path.join(getattr(params, "data_dir", ""), "*valid*")
        is_reversed = getattr(params, 'is_reversed')

        dataset = self._read_and_batch_from_tfrecord(
            file_pattern, params.batch_size, params.max_length,
            params.num_parallel_calls, shuffle=False, repeat=1, is_reversed=is_reversed)
        
        #return dataset
        # iterator = dataset.make_one_shot_iterator()
        iterator = dataset.make_initializable_iterator()
        src, tgt = iterator.get_next()
        return BatchedInput(initializer=iterator.initializer,
                            source=src,
                            target=tgt)

if __name__ == "__main__":
    class Params(object):
        data_dir_mono = "/home/work/xiepan/xp_dial/gan_nmt/transformer_rl/data/en-tr/v1_mono/gen_data"
        data_dir = "/home/work/xiepan/xp_dial/gan_nmt/transformer_rl/data/en-tr/v1/gen_data"
        is_reversed = False
        batch_size = 500  # 这里的batch_size 不是最终的batch_size,这个值必须大于 max_length
        max_length = 50
        num_parallel_calls = 6
        repeat_dataset = 1

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    #params = Params()
    #tf.enable_eager_execution()
    #dataset = Dataset(params)
    #train_dataset = dataset.train_input_fn(params)
    #for batch, content in enumerate(train_dataset):
    #    if batch > 50:
    #        break
    #    print(content[0].shape)
    params = Params()
    dataset = Dataset(params)
    input_fn = dataset.train_input_fn_mono(params)
    #iterator = train_dataset.make_initializable_iterator()
    #batch = iterator.get_next()

    with tf.Session() as sess:
        sess.run(input_fn.initializer)
        while True:
            try:
                src, tgt = sess.run([input_fn.source, input_fn.target])
                print("src, ", src[:2])
                print("tgt, ", tgt.shape)
            except tf.errors.OutOfRangeError:
                break
