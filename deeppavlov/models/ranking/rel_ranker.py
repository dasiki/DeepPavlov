import numpy as np
import tensorflow as tf
from typing import List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import LRScheduledTFModel
from deeppavlov.core.models.component import Component
from deeppavlov.models.embedders.abstract_embedder import Embedder
from deeppavlov.core.layers.tf_layers import cudnn_bi_gru, variational_dropout
from deeppavlov.models.squad.utils import softmax_mask


@register('two_sentences_emb')
class TwoSentencesEmbedder(Component):
    """
        This class is used for embedding of two sentences
    """

    def __init__(self, embedder: Embedder, **kwargs):
        """

        Args:
            embedder: what embedder to use: Glove, Fasttext or other
            **kwargs:
        """
        self.embedder = embedder

    def __call__(self, sentence_tokens_1: List[List[str]], sentence_tokens_2: List[List[str]]):
        sentence_token_embs_1 = self.embedder(sentence_tokens_1)
        sentence_token_embs_2 = self.embedder(sentence_tokens_2)
        return sentence_token_embs_1, sentence_token_embs_2


@register('rel_ranker')
class RelRanker(LRScheduledTFModel):
    """
        This class determines is the relation appropriate for the question or not
    """

    def __init__(self, n_classes: int = 2,
                 dropout_keep_prob: float = 0.5,
                 return_probas: bool = False, **kwargs):
        """

        Args:
            n_classes: number of classes for classification
            dropout_keep_prob: Probability of keeping the hidden state, values from 0 to 1. 0.5 works well in most cases.
            return_probas: whether to return confidences of the relation to be appropriate or not
            **kwargs:
        """
        if 'learning_rate_drop_div' not in kwargs:
            kwargs['learning_rate_drop_div'] = 10.0
        if 'learning_rate_drop_patience' not in kwargs:
            kwargs['learning_rate_drop_patience'] = 5.0
        if 'clip_norm' not in kwargs:
            kwargs['clip_norm'] = 5.0

        super().__init__(**kwargs)

        self.n_classes = n_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.return_probas = return_probas
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.question_ph = tf.placeholder(tf.float32, [None, None, 300])
        self.rel_emb_ph = tf.placeholder(tf.float32, [None, None, 300])

        r_mask_2 = tf.cast(self.rel_emb_ph, tf.bool)
        r_len_2 = tf.reduce_sum(tf.cast(r_mask_2, tf.int32), axis=2)
        r_mask = tf.cast(r_len_2, tf.bool)
        r_len = tf.reduce_sum(tf.cast(r_mask, tf.int32), axis=1)

        self.y_ph = tf.placeholder(tf.int32, shape=(None,))
        self.one_hot_labels = tf.one_hot(self.y_ph, depth=self.n_classes, dtype=tf.float32)
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')

        q_mask_2 = tf.cast(self.question_ph, tf.bool)
        q_len_2 = tf.reduce_sum(tf.cast(q_mask_2, tf.int32), axis=2)
        q_mask = tf.cast(q_len_2, tf.bool)
        q_len = tf.reduce_sum(tf.cast(q_mask, tf.int32), axis=1)

        question_dr = variational_dropout(self.question_ph, keep_prob=self.keep_prob_ph)

        with tf.variable_scope("question_encode"):
            (h_fw, h_bw), _ = cudnn_bi_gru(question_dr, 150, seq_lengths=q_len)
            q = tf.concat([h_fw, h_bw], axis=2)
        with tf.variable_scope("relation_encode"):
            _, (h_last_fw, h_last_bw) = cudnn_bi_gru(self.rel_emb_ph, 150, seq_lengths=r_len)
            r = tf.concat([h_last_fw, h_last_bw], axis=1)
            r_exp = tf.expand_dims(r, axis=1)
        with tf.variable_scope("attention"):
            dot_products = tf.reduce_sum(tf.multiply(q, r_exp), axis=2, keep_dims=False)
            s_mask = softmax_mask(dot_products, q_mask)
            att_weights = tf.expand_dims(tf.nn.softmax(s_mask), axis=2)
            self.s_r = tf.reduce_sum(tf.multiply(att_weights, q), axis=1)

            self.logits = tf.layers.dense(tf.multiply(self.s_r, r), 2, activation=None, use_bias=False)
            self.y_pred = tf.argmax(self.logits, axis=-1)

            loss_tensor = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.one_hot_labels, logits=self.logits)

            self.loss = tf.reduce_mean(loss_tensor)
            self.train_op = self.get_train_op(self.loss)

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.load()

    def fill_feed_dict(self, xs, y=None, train=False):
        xs = list(xs)
        xs[0] = np.array(xs[0])
        xs[1] = np.array(xs[1])
        feed_dict = {self.question_ph: xs[0], self.rel_emb_ph: xs[1]}
        if y is not None:
            feed_dict[self.y_ph] = y
        if train:
            feed_dict[self.keep_prob_ph] = self.dropout_keep_prob
        if not train:
            feed_dict[self.keep_prob_ph] = 1.0

        return feed_dict

    def predict(self, xs):
        feed_dict = self.fill_feed_dict(xs)
        if self.return_probas:
            pred = self.sess.run(self.logits, feed_dict)
        else:
            pred = self.sess.run(self.y_pred, feed_dict)
        return pred

    def __call__(self, *args, **kwargs):
        if len(args[0]) == 0 or (len(args[0]) == 1 and len(args[0][0]) == 0):
            return []
        return self.predict(args)

    def train_on_batch(self, *args):
        *xs, y = args
        feed_dict = self.fill_feed_dict(xs, y, train=True)
        _, loss_value = self.sess.run([self.train_op, self.loss], feed_dict)

        return {'loss': loss_value,
                'learning_rate': self.get_learning_rate(),
                'momentum': self.get_momentum()}

    def process_event(self, event_name, data):
        super().process_event(event_name, data)
