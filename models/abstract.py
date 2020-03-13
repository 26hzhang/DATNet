import os
import codecs
import json
import math
import random
import numpy as np
import tensorflow as tf
from utils.logger import get_logger
from utils.CoNLLeval import CoNLLeval


class Abstract:
    def __init__(self, config):
        self.cfg = config
        self.checkpoint_path = "ckpt/{}/".format(self.cfg.model_name)
        tf.set_random_seed(self.cfg.random_seed)
        # create folders and logger
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.logger = get_logger(os.path.join(self.checkpoint_path, "log.txt"))

    def _initialize_session(self):
        if not self.cfg.use_gpu:
            self.sess = tf.Session()
        else:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=self.cfg.max_to_keep)
        self.sess.run(tf.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        self.saver.save(self.sess, self.checkpoint_path + self.cfg.model_name, global_step=epoch)

    def close_session(self):
        self.sess.close()

    @staticmethod
    def count_params(scope=None):
        if scope is None:
            return int(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        else:
            return int(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables(scope)]))

    @staticmethod
    def load_dataset(filename):
        with codecs.open(filename, mode='r', encoding='utf-8') as f:
            dataset = json.load(f)
        return dataset

    def _add_summary(self, summary_path):
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(summary_path + "train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(summary_path + "test")

    @staticmethod
    def _arrange_batches(src_num_batches, tgt_num_batches, mix_rate):
        num_src = int(src_num_batches * mix_rate)
        num_tgt = math.ceil(tgt_num_batches * 0.8)
        batches = ["src"] * num_src + ["tgt"] * num_tgt
        random.shuffle(batches)
        batches += ["tgt"] * (tgt_num_batches - num_tgt)
        return batches

    def evaluate_f1(self, dataset, rev_word_dict, rev_label_dict, name):
        save_path = os.path.join(self.checkpoint_path, name + "_result.txt")
        if os.path.exists(save_path):
            os.remove(save_path)
        predictions, groundtruth, words_list = list(), list(), list()
        for b_labels, b_predicts, b_words, b_seq_len in dataset:
            for labels, predicts, words, seq_len in zip(b_labels, b_predicts, b_words, b_seq_len):
                predictions.append([rev_label_dict[x] for x in predicts[:seq_len]])
                groundtruth.append([rev_label_dict[x] for x in labels[:seq_len]])
                words_list.append([rev_word_dict[x] for x in words[:seq_len]])
        conll_eval = CoNLLeval()
        score = conll_eval.conlleval(predictions, groundtruth, words_list, save_path)
        self.logger.info("{} dataset -- acc: {:04.2f}, pre: {:04.2f}, rec: {:04.2f}, FB1: {:04.2f}"
                         .format(name, score["accuracy"], score["precision"], score["recall"], score["FB1"]))
        return score["FB1"]

    def evaluate_acc(self, dataset, name):
        corrects, total = 0, 0
        for b_labels, b_predicts, _, b_seq_len in dataset:
            for labels, predicts, seq_len in zip(b_labels, b_predicts, b_seq_len):
                corrects += sum([1 if x == y else 0 for x, y in zip(predicts[:seq_len], labels[:seq_len])])
                total += seq_len
        accuracy = float(corrects) / float(total) * 100.0
        self.logger.info("{} dataset -- accuracy: {:04.2f}".format(name, accuracy))
        return accuracy

    def evaluate(self, dataset, name, task, predict_op, rev_word_dict, rev_label_dict):
        all_data = list()
        for data in dataset:
            predicts = predict_op(data)
            all_data.append((data["labels"], predicts, data["words"], data["seq_len"]))
        if "pos" in task:
            return self.evaluate_acc(all_data, name)
        else:
            return self.evaluate_f1(all_data, rev_word_dict, rev_label_dict, name)
