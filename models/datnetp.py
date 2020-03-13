import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from utils.logger import Progbar
from models.abstract import Abstract
from models.modules import create_optimizer, viterbi_decode
from models.modules import char_cnn_hw, bi_rnn, crf_layer, discriminator, embedding_lookup, add_perturbation


class DATNetPModel(Abstract):
    """Part Transfer and DATNet-P modules"""
    def __init__(self, config, vocab):
        super(DATNetPModel, self).__init__(config)
        self.vocab = vocab
        with tf.Graph().as_default():
            self._add_placeholders()
            self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True) if self.cfg.elmo else None
            self._build_model()
            self.logger.info("total params: {}".format(self.count_params()))
            self._initialize_session()

    def _get_elmo_emb(self, words, seq_len):
        emb = self.elmo(inputs={"tokens": words, "sequence_len": seq_len}, signature="tokens", as_dict=True)["elmo"]
        return emb

    def _get_feed_dict(self, src_data, tgt_data, domain_labels, training=False):
        feed_dict = {self.training: training}
        if src_data is not None:
            if self.cfg.elmo:
                feed_dict[self.src_words] = src_data["words_str"]
            else:
                feed_dict[self.src_words] = src_data["words"]
            feed_dict[self.src_seq_len] = src_data["seq_len"]
            feed_dict[self.src_chars] = src_data["chars"]
            feed_dict[self.src_char_seq_len] = src_data["char_seq_len"]
            if "labels" in src_data:
                feed_dict[self.src_labels] = src_data["labels"]
        if tgt_data is not None:
            if self.cfg.elmo:
                feed_dict[self.tgt_words] = tgt_data["words_str"]
            else:
                feed_dict[self.tgt_words] = tgt_data["words"]
            feed_dict[self.tgt_seq_len] = tgt_data["seq_len"]
            feed_dict[self.tgt_chars] = tgt_data["chars"]
            feed_dict[self.tgt_char_seq_len] = tgt_data["char_seq_len"]
            if "labels" in tgt_data:
                feed_dict[self.tgt_labels] = tgt_data["labels"]
        if domain_labels is not None:
            feed_dict[self.domain_labels] = domain_labels
        return feed_dict

    def _add_placeholders(self):
        # source placeholders
        if self.cfg.elmo:
            self.src_words = tf.placeholder(tf.string, shape=[None, None], name="source_words")
        else:
            self.src_words = tf.placeholder(tf.int32, shape=[None, None], name="source_words")
        self.src_seq_len = tf.placeholder(tf.int32, shape=[None], name="source_seq_len")
        self.src_chars = tf.placeholder(tf.int32, shape=[None, None, None], name="source_chars")
        self.src_char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="source_char_seq_len")
        self.src_labels = tf.placeholder(tf.int32, shape=[None, None], name="source_labels")
        # target placeholders
        if self.cfg.elmo:
            self.tgt_words = tf.placeholder(tf.string, shape=[None, None], name="target_words")
        else:
            self.tgt_words = tf.placeholder(tf.int32, shape=[None, None], name="target_words")
        self.tgt_seq_len = tf.placeholder(tf.int32, shape=[None], name="target_seq_len")
        self.tgt_chars = tf.placeholder(tf.int32, shape=[None, None, None], name="target_chars")
        self.tgt_char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="target_char_seq_len")
        self.tgt_labels = tf.placeholder(tf.int32, shape=[None, None], name="target_labels")
        # domain labels
        self.domain_labels = tf.placeholder(tf.int32, shape=[None, 2], name="domain_labels")
        # hyper-parameters
        self.training = tf.placeholder(tf.bool, shape=[], name="training")

    def _build_model(self):
        if self.cfg.elmo:
            src_word_emb = self._get_elmo_emb(self.src_words, self.src_seq_len)
            tgt_word_emb = self._get_elmo_emb(self.tgt_words, self.tgt_seq_len)
        else:
            if self.cfg.share_word:
                src_word_emb = embedding_lookup(self.src_words, self.vocab.src_word_size, self.cfg.src_word_dim,
                                                self.vocab.src_wordvec, self.vocab.src_word_weight, self.cfg.tune_emb,
                                                self.cfg.norm_emb, self.cfg.word_project, None, self.cfg.at,
                                                name="word_table")
                tgt_word_emb = embedding_lookup(self.tgt_words, self.vocab.src_word_size, self.cfg.src_word_dim,
                                                self.vocab.src_wordvec, self.vocab.src_word_weight, self.cfg.tune_emb,
                                                self.cfg.norm_emb, self.cfg.word_project, None, self.cfg.at,
                                                name="word_table")
            else:
                src_word_emb = embedding_lookup(self.src_words, self.vocab.src_word_size, self.cfg.src_word_dim,
                                                self.vocab.src_wordvec, self.vocab.src_word_weight, self.cfg.tune_emb,
                                                self.cfg.norm_emb, self.cfg.word_project, None, self.cfg.at,
                                                name="src_word_table")
                tgt_word_emb = embedding_lookup(self.tgt_words, self.vocab.tgt_word_size, self.cfg.tgt_word_dim,
                                                self.vocab.tgt_wordvec, self.vocab.tgt_word_weight, self.cfg.tune_emb,
                                                self.cfg.norm_emb, self.cfg.word_project, None, self.cfg.at,
                                                name="tgt_word_table")

        src_char_emb = embedding_lookup(self.src_chars, self.vocab.char_size, self.cfg.char_dim, None,
                                        self.vocab.char_weight, True, self.cfg.norm_emb, False, None, self.cfg.at,
                                        name="char_table")
        tgt_char_emb = embedding_lookup(self.tgt_chars, self.vocab.char_size, self.cfg.char_dim, None,
                                        self.vocab.char_weight, True, self.cfg.norm_emb, False, None, self.cfg.at,
                                        name="char_table")

        def compute_src_logits(w_emb, c_emb):
            char_cnn = char_cnn_hw(c_emb, self.cfg.kernel_sizes, self.cfg.filters, self.cfg.char_dim, self.cfg.hw_layer,
                                   activation=tf.tanh, name="char_cnn_hw")
            emb = tf.layers.dropout(tf.concat([w_emb, char_cnn], axis=-1), rate=self.cfg.emb_drop_rate,
                                    training=self.training)
            rnn_feats = bi_rnn(emb, self.src_seq_len, self.training, self.cfg.src_num_units, self.cfg.rnn_drop_rate,
                               activation=tf.tanh, concat=self.cfg.concat_rnn, name="src_birnn")
            share_rnn_feats = bi_rnn(emb, self.src_seq_len, self.training, self.cfg.share_num_units,
                                     self.cfg.rnn_drop_rate, activation=tf.tanh, concat=self.cfg.concat_rnn,
                                     name="share_birnn")
            rnn_feats = tf.concat([rnn_feats, share_rnn_feats], axis=-1)
            logits = tf.layers.dense(rnn_feats, units=self.vocab.src_label_size, reuse=tf.AUTO_REUSE, name="src_proj")
            transition, crf_loss = crf_layer(logits, self.src_labels, self.src_seq_len, self.vocab.src_label_size,
                                             name="crf" if self.cfg.share_label else "src_crf")
            dis_loss = discriminator(rnn_feats, self.domain_labels, 2, self.cfg.grad_rev_rate, self.cfg.alpha,
                                     self.cfg.gamma, self.cfg.disc, name="discriminator")
            if dis_loss is not None:
                crf_loss = crf_loss + dis_loss
            return logits, transition, crf_loss

        def compute_tgt_logits(w_emb, c_emb):
            char_cnn = char_cnn_hw(c_emb, self.cfg.kernel_sizes, self.cfg.filters, self.cfg.char_dim, self.cfg.hw_layer,
                                   activation=tf.tanh, name="char_cnn_hw")
            emb = tf.layers.dropout(tf.concat([w_emb, char_cnn], axis=-1), rate=self.cfg.emb_drop_rate,
                                    training=self.training)
            rnn_feats = bi_rnn(emb, self.tgt_seq_len, self.training, self.cfg.tgt_num_units, self.cfg.rnn_drop_rate,
                               activation=tf.tanh, concat=self.cfg.concat_rnn, name="tgt_birnn")
            share_rnn_feats = bi_rnn(emb, self.tgt_seq_len, self.training, self.cfg.share_num_units,
                                     self.cfg.rnn_drop_rate, activation=tf.tanh, concat=self.cfg.concat_rnn,
                                     name="share_birnn")
            rnn_feats = tf.concat([rnn_feats, share_rnn_feats], axis=-1)
            logits = tf.layers.dense(rnn_feats, units=self.vocab.tgt_label_size, reuse=tf.AUTO_REUSE, name="tgt_proj")
            transition, crf_loss = crf_layer(logits, self.tgt_labels, self.tgt_seq_len, self.vocab.tgt_label_size,
                                             name="crf" if self.cfg.share_label else "tgt_crf")
            dis_loss = discriminator(rnn_feats, self.domain_labels, 2, self.cfg.grad_rev_rate, self.cfg.alpha,
                                     self.cfg.gamma, self.cfg.disc, name="discriminator")
            if dis_loss is not None:
                crf_loss = crf_loss + dis_loss
            return logits, transition, crf_loss

        # train source
        self.src_logits, self.src_transition, self.src_loss = compute_src_logits(src_word_emb, src_char_emb)
        if self.cfg.at:  # adversarial training
            perturb_src_word_emb = add_perturbation(src_word_emb, self.src_loss, epsilon=self.cfg.epsilon)
            perturb_src_char_emb = add_perturbation(src_char_emb, self.src_loss, epsilon=self.cfg.epsilon)
            *_, adv_src_loss = compute_src_logits(perturb_src_word_emb, perturb_src_char_emb)
            self.src_loss = self.src_loss + adv_src_loss

        # train target
        self.tgt_logits, self.tgt_transition, self.tgt_loss = compute_tgt_logits(tgt_word_emb, tgt_char_emb)
        if self.cfg.at:  # adversarial training
            perturb_tgt_word_emb = add_perturbation(tgt_word_emb, self.tgt_loss, epsilon=self.cfg.epsilon)
            perturb_tgt_char_emb = add_perturbation(tgt_char_emb, self.tgt_loss, epsilon=self.cfg.epsilon)
            *_, adv_tgt_loss = compute_tgt_logits(perturb_tgt_word_emb, perturb_tgt_char_emb)
            self.tgt_loss = self.tgt_loss + adv_tgt_loss

        self.src_train_op = create_optimizer(self.src_loss, self.cfg.lr, self.cfg.decay_step,
                                             opt_name=self.cfg.optimizer, grad_clip=self.cfg.grad_clip, name="src_opt")
        self.tgt_train_op = create_optimizer(self.tgt_loss, self.cfg.lr, self.cfg.decay_step,
                                             opt_name=self.cfg.optimizer, grad_clip=self.cfg.grad_clip, name="tgt_opt")

    def _src_predict_op(self, data):
        feed_dict = self._get_feed_dict(src_data=data, tgt_data=None, domain_labels=None)
        logits, transition, seq_len = self.sess.run([self.src_logits, self.src_transition, self.src_seq_len],
                                                    feed_dict=feed_dict)
        return viterbi_decode(logits, transition, seq_len)

    def _tgt_predict_op(self, data):
        feed_dict = self._get_feed_dict(src_data=None, tgt_data=data, domain_labels=None)
        logits, transition, seq_len = self.sess.run([self.tgt_logits, self.tgt_transition, self.tgt_seq_len],
                                                    feed_dict=feed_dict)
        return viterbi_decode(logits, transition, seq_len)

    def train(self, src_dataset, tgt_dataset):
        self.logger.info("Start training...")
        best_score, no_imprv_epoch, src_lr, tgt_lr, cur_step = -np.inf, 0, self.cfg.lr, self.cfg.lr, 0
        for epoch in range(1, self.cfg.epochs + 1):
            self.logger.info("Epoch {}/{}:".format(epoch, self.cfg.epochs))
            batches = self._arrange_batches(src_dataset.num_batches, tgt_dataset.num_batches, self.cfg.mix_rate)
            prog = Progbar(target=len(batches))
            prog.update(0, [("Global Step", int(cur_step)), ("Source Train Loss", 0.0), ("Target Train Loss", 0.0)])
            for i, batch_name in enumerate(batches):
                cur_step += 1
                if batch_name == "src":
                    data = src_dataset.next_batch()
                    domain_labels = [[1, 0]] * data["batch_size"]
                    feed_dict = self._get_feed_dict(src_data=data, tgt_data=None, domain_labels=domain_labels,
                                                    training=True)
                    _, src_cost = self.sess.run([self.src_train_op, self.src_loss], feed_dict=feed_dict)
                    prog.update(i + 1, [("Global Step", int(cur_step)), ("Source Train Loss", src_cost)])
                else:  # "tgt"
                    data = tgt_dataset.next_batch()
                    domain_labels = [[0, 1]] * data["batch_size"]
                    feed_dict = self._get_feed_dict(src_data=None, tgt_data=data, domain_labels=domain_labels,
                                                    training=True)
                    _, tgt_cost = self.sess.run([self.tgt_train_op, self.tgt_loss], feed_dict=feed_dict)
                    prog.update(i + 1, [("Global Step", int(cur_step)), ("Target Train Loss", tgt_cost)])
            score = self.evaluate_data(tgt_dataset.dev_batches(), "target_dev", resource="target")
            self.evaluate_data(tgt_dataset.test_batches(), "target_test", resource="target")
            if score > best_score:
                best_score, no_imprv_epoch = score, 0
                self.save_session(epoch)
                self.logger.info(' -- new BEST score on target dev dataset: {:04.2f}'.format(best_score))
            else:
                no_imprv_epoch += 1
                if self.cfg.no_imprv_tolerance is not None and no_imprv_epoch >= self.cfg.no_imprv_tolerance:
                    self.logger.info('early stop at {}th epoch without improvement'.format(epoch))
                    self.logger.info('best score on target dev set: {}'.format(best_score))
                    break

    def evaluate_data(self, dataset, name, resource="target"):
        if resource == "target":
            score = self.evaluate(dataset, name, self.cfg.tgt_task, self._tgt_predict_op, self.vocab.tgt_id_to_word,
                                  self.vocab.tgt_id_to_label)
        else:
            score = self.evaluate(dataset, name, self.cfg.src_task, self._src_predict_op, self.vocab.src_id_to_word,
                                  self.vocab.src_id_to_label)
        return score
