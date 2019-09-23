"""Functions for calculating loss, accuracy, and other model metrics.

Metrics:
 - Padded loss, accuracy, and negative log perplexity. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/metrics.py
 - BLEU approximation. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py
 - ROUGE score. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/rouge.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from utils.tokenizer import EOS_ID, PAD_ID

def _convert_to_eval_metric(metric_fn):
    """Wrap a metric fn that returns scores and weights as an eval metric fn.

    The input metric_fn returns values for the current batch. The wrapper
    aggregates the return values collected over all of the batches evaluated.

    Args:
      metric_fn: function that returns scores and weights for the current batch's
        logits and predicted labels.

    Returns:
      function that aggregates the scores and weights from metric_fn.
    """

    def problem_metric_fn(*args):
        """Returns an aggregation of the metric_fn's returned values."""
        (scores, weights) = metric_fn(*args)

        # The tf.metrics.mean function assures correct aggregation.
        return tf.metrics.mean(scores, weights)

    return problem_metric_fn


def get_eval_metrics(logits, labels, params):
    """Return dictionary of model evaluation metrics."""
    metrics = {
        # "accuracy": _convert_to_eval_metric(padded_accuracy)(logits, labels),
        # "accuracy_top5": _convert_to_eval_metric(padded_accuracy_top5)(
        #     logits, labels),
        # "accuracy_per_sequence": _convert_to_eval_metric(
        #     padded_sequence_accuracy)(logits, labels),
        # "neg_log_perplexity": _convert_to_eval_metric(padded_neg_log_perplexity)(
        #     logits, labels, params.target_vocab_size),
        "approx_bleu_score": _convert_to_eval_metric(bleu_score)(logits, labels),
        # "rouge_2_fscore": _convert_to_eval_metric(rouge_2_fscore)(logits, labels),
        # "rouge_L_fscore": _convert_to_eval_metric(rouge_l_fscore)(logits, labels),
    }

    # Prefix each of the metric names with "metrics/". This allows the metric
    # graphs to display under the "metrics" category in TensorBoard.
    metrics = {"metrics/%s" % k: v for k, v in six.iteritems(metrics)}
    return metrics

def get_eval_metrics_rl(predictions, labels, params):
    """Return dictionary of model evaluation metrics."""
    metrics = {
        # "accuracy": _convert_to_eval_metric(padded_accuracy)(logits, labels),
        # "accuracy_top5": _convert_to_eval_metric(padded_accuracy_top5)(
        #     logits, labels),
        # "accuracy_per_sequence": _convert_to_eval_metric(
        #     padded_sequence_accuracy)(logits, labels),
        # "neg_log_perplexity": _convert_to_eval_metric(padded_neg_log_perplexity)(
        #     logits, labels, params.target_vocab_size),
        # "approx_bleu_score": _convert_to_eval_metric(bleu_score)(logits, labels),
        "approx_bleu_rl_score": _convert_to_eval_metric(bleu_score_rl)(predictions, labels)
        # "rouge_2_fscore": _convert_to_eval_metric(rouge_2_fscore)(logits, labels),
        # "rouge_L_fscore": _convert_to_eval_metric(rouge_l_fscore)(logits, labels),
    }

    # Prefix each of the metric names with "metrics/". This allows the metric
    # graphs to display under the "metrics" category in TensorBoard.
    metrics = {"metrics/%s" % k: v for k, v in six.iteritems(metrics)}
    return metrics


def _pad_tensors_to_same_length(x, y):
    """Pad x and y so that the results have the same length (second dimension)."""
    with tf.name_scope("pad_to_same_length"):
        x_length = tf.shape(x)[1]
        y_length = tf.shape(y)[1]

        max_length = tf.maximum(x_length, y_length)

        x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
        return x, y


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
    """Calculate cross entropy loss while ignoring padding.

    Args:
      logits: Tensor of size [batch_size, length_logits, vocab_size]
      labels: Tensor of size [batch_size, length_labels]
      smoothing: Label smoothing constant, used to determine the on and off values
      vocab_size: int size of the vocabulary
    Returns:
      Returns the cross entropy loss and weight tensors: float32 tensors with
        shape [batch_size, max(length_logits, length_labels)]
    """
    with tf.name_scope("loss"):
        logits, labels = _pad_tensors_to_same_length(logits, labels)

        # Calculate smoothing cross entropy
        with tf.name_scope("smoothing_cross_entropy"):
            confidence = 1.0 - smoothing
            low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
            soft_targets = tf.one_hot(
                tf.cast(labels, tf.int32),
                depth=vocab_size,
                on_value=confidence,
                off_value=low_confidence)
            xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=soft_targets)

            # Calculate the best (lowest) possible value of cross entropy, and
            # subtract from the cross entropy loss.
            normalizing_constant = -(
                    confidence * tf.log(confidence) + tf.to_float(vocab_size - 1) *
                    low_confidence * tf.log(low_confidence + 1e-20))
            xentropy -= normalizing_constant

        weights = tf.to_float(tf.not_equal(labels, 0))
        return xentropy * weights, weights


def bleu_score(logits, labels):
    """Approximate BLEU score computation between labels and predictions.

    An approximate BLEU scoring method since we do not glue word pieces or
    decode the ids and tokenize the output. By default, we use ngram order of 4
    and use brevity penalty. Also, this does not have beam search.

    Args:
      logits: Tensor of size [batch_size, length_logits, vocab_size]
      labels: Tensor of size [batch-size, length_labels]

    Returns:
      bleu: int, approx bleu score
    """
    predictions = tf.to_int32(tf.argmax(logits, axis=-1))
    # TODO: Look into removing use of py_func
    bleu = tf.py_func(compute_bleu, (labels, predictions), tf.float32)
    return bleu, tf.constant(1.0)

def bleu_score_rl(predictions, labels):
    """Approximate BLEU score computation between labels and predictions.

    An approximate BLEU scoring method since we do not glue word pieces or
    decode the ids and tokenize the output. By default, we use ngram order of 4
    and use brevity penalty. Also, this does not have beam search.

    Args:
      predictions: [batch, gen_len]
      labels: Tensor of size [batch-size, length_labels]

    Returns:
      bleu: int, approx bleu score
    """
    # predictions = tf.to_int32(tf.argmax(logits, axis=-1))
    # TODO: Look into removing use of py_func
    bleu = tf.py_func(compute_bleu, (labels, predictions), tf.float32)
    return bleu, tf.constant(1.0)


def _get_ngrams_with_counter(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.

    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in xrange(1, max_order + 1):
        for i in xrange(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts

def _trim(ids):
    """Trim EOS and PAD tokens from ids, and decode to return a string."""
    try:
        index = list(ids).index(EOS_ID)
        #tf.logging.info(ids[:index])
        return ids[:index]
    except ValueError:  # No EOS found in sequence
        return ids

def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 use_bp=True):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      use_bp: boolean, whether to apply brevity penalty.

    Returns:
      BLEU score.
    """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order

    for (references, translations) in zip(reference_corpus, translation_corpus):
        references = _trim(references)
        translations = _trim(translations)

        reference_length += len(references)
        translation_length += len(translations)
        ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
        translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

        overlap = dict((ngram,
                        min(count, translation_ngram_counts[ngram]))
                       for ngram, count in ref_ngram_counts.items())

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[
                ngram]

    precisions = [0] * max_order
    smooth = 1.0

    for i in xrange(0, max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
            if matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[
                    i]
            else:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
        else:
            precisions[i] = 0.0

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum / max_order)

    if use_bp:
        ratio = translation_length / reference_length
        bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
    bleu = geo_mean * bp
    return np.float32(bleu)


def compute_bleu_batch(reference_corpus, translation_corpus, max_order=4,
                 use_bp=True):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      use_bp: boolean, whether to apply brevity penalty.

    Returns:
      BLEU score.
    """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_order_batch = []
    possible_matches_by_order_batch = []

    for (references, translations) in zip(reference_corpus, translation_corpus):
        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order

        reference_length += len(references)
        translation_length += len(translations)
        ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
        translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

        overlap = dict((ngram,
                        min(count, translation_ngram_counts[ngram]))
                       for ngram, count in ref_ngram_counts.items())

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[
                ngram]
        matches_by_order_batch.append(matches_by_order)
        possible_matches_by_order_batch.append(possible_matches_by_order)

    smooth = 1.0
    bleu_batch = []
    for idx in xrange(len(matches_by_order_batch)):
        precisions = [0] * max_order
        for i in xrange(0, max_order):
            if possible_matches_by_order_batch[idx][i] > 0:
                precisions[i] = float(matches_by_order_batch[idx][i]) / possible_matches_by_order_batch[idx][i]
                if matches_by_order_batch[idx][i] > 0:
                    precisions[i] = float(matches_by_order_batch[idx][i]) / possible_matches_by_order_batch[idx][i]
                else:
                    smooth *= 2
                    precisions[i] = 1.0 / (smooth * possible_matches_by_order_batch[idx][i])
            else:
                precisions[i] = 0.0
        if max(precisions) > 0:
            p_log_sum = sum(math.log(p) for p in precisions if p)
            geo_mean = math.exp(p_log_sum / max_order)

        if use_bp:
            ratio = translation_length / reference_length
            bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
        bleu = geo_mean * bp
        bleu_batch.append(bleu)
    return np.float32(bleu_batch)

if __name__ == "__main__":
    tf.enable_eager_execution()
    logits = tf.random_normal((2, 2, 100), dtype=tf.float32)
    labels = tf.constant([[1,2],[3,4]], dtype=tf.int32)
    out, weights = padded_cross_entropy_loss(logits, labels, 0.1, 100)
    print(out, weights)
