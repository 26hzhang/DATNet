import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from utils.logger import Progbar
from models.abstract import Abstract
from models.modules import create_optimizer, viterbi_decode
from models.modules import bi_rnn, char_cnn_hw, crf_layer, embedding_lookup, add_perturbation


class BaseModel(Abstract):
    """Basement model: Char CNN + LSTM encoder + Chain CRF decoder module"""
    def __init__(self, config, vocab):
        super(BaseModel, self).__init__(config)
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

    def _get_feed_dict(self, data, training=False):
        feed_dict = {self.seq_len: data["seq_len"], self.chars: data["chars"], self.char_seq_len: data["char_seq_len"]}
        if self.cfg.elmo:
            feed_dict[self.words] = data["words_str"]
        else:
            feed_dict[self.words] = data["words"]
        if "labels" in data:
            feed_dict[self.labels] = data["labels"]
        feed_dict[self.training] = training
        return feed_dict

    def _add_placeholders(self):
        if self.cfg.elmo:
            self.words = tf.placeholder(tf.string, shape=[None, None], name="words")
        else:
            self.words = tf.placeholder(tf.int32, shape=[None, None], name="words")
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
        self.chars = tf.placeholder(tf.int32, shape=[None, None, None], name="chars")
        self.char_seq_len = tf.placeholder(tf.int32, shape=[None, None], name="char_seq_len")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.training = tf.placeholder(tf.bool, shape=[], name="training")

    def _build_model(self):
        if self.cfg.elmo:
            word_emb = self._get_elmo_emb(self.words, self.seq_len)
        else:
            word_emb = embedding_lookup(self.words, self.vocab.word_size, self.cfg.word_dim, self.vocab.wordvec,
                                        self.vocab.word_weight, self.cfg.tune_emb, self.cfg.norm_emb,
                                        self.cfg.word_project, None, self.cfg.at, name="word_table")
        char_emb = embedding_lookup(self.chars, self.vocab.char_size, self.cfg.char_dim, None, self.vocab.char_weight,
                                    True, self.cfg.norm_emb, False, None, self.cfg.at, name="char_table")

        def compute_logits(w_emb, c_emb):
            char_cnn = char_cnn_hw(c_emb, self.cfg.kernel_sizes, self.cfg.filters, self.cfg.char_dim, self.cfg.hw_layer,
                                   activation=tf.tanh, name="char_cnn_hw")
            emb = tf.layers.dropout(tf.concat([w_emb, char_cnn], axis=-1), rate=self.cfg.emb_drop_rate,
                                    training=self.training)
            rnn_feats = bi_rnn(emb, self.seq_len, self.training, self.cfg.num_units, self.cfg.rnn_drop_rate,
                               activation=tf.tanh, concat=self.cfg.concat_rnn, name="bi_rnn")
            logits = tf.layers.dense(rnn_feats, units=self.vocab.label_size, reuse=tf.AUTO_REUSE, name="project")
            transition, crf_loss = crf_layer(logits, self.labels, self.seq_len, self.vocab.label_size, name="crf")
            return logits, transition, crf_loss

        self.logits, self.transition, self.loss = compute_logits(word_emb, char_emb)
        if self.cfg.at:
            perturb_word_emb = add_perturbation(word_emb, self.loss, epsilon=self.cfg.epsilon)
            perturb_char_emb = add_perturbation(char_emb, self.loss, epsilon=self.cfg.epsilon)
            *_, adv_loss = compute_logits(perturb_word_emb, perturb_char_emb)
            self.loss += adv_loss

        self.train_op = create_optimizer(self.loss, self.cfg.lr, self.cfg.decay_step, opt_name=self.cfg.optimizer,
                                         grad_clip=self.cfg.grad_clip, name="optimizer")

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        logits, transition, seq_len = self.sess.run([self.logits, self.transition, self.seq_len], feed_dict=feed_dict)
        return viterbi_decode(logits, transition, seq_len)

    def train(self, dataset):
        self.logger.info("Start training...")
        best_score, no_imprv_epoch, cur_step = -np.inf, 0, 0
        for epoch in range(1, self.cfg.epochs + 1):
            self.logger.info('Epoch {}/{}:'.format(epoch, self.cfg.epochs))
            prog = Progbar(target=dataset.num_batches)
            for i in range(dataset.num_batches):
                cur_step += 1
                data = dataset.next_batch()
                feed_dict = self._get_feed_dict(data, training=True)
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss)])
            # evaluate
            score = self.evaluate_data(dataset.dev_batches(), name="dev")
            self.evaluate_data(dataset.test_batches(), name="test")
            if score > best_score:
                best_score, no_imprv_epoch = score, 0
                self.save_session(epoch)
                self.logger.info(' -- new BEST score on dev dataset: {:04.2f}'.format(best_score))
            else:
                no_imprv_epoch += 1
                if self.cfg.no_imprv_tolerance is not None and no_imprv_epoch >= self.cfg.no_imprv_tolerance:
                    self.logger.info('early stop at {}th epoch without improvement'.format(epoch))
                    self.logger.info('best score on dev set: {}'.format(best_score))
                    break

    def evaluate_data(self, dataset, name):
        score = self.evaluate(dataset, name, self.cfg.task, self._predict_op, self.vocab.id_to_word,
                              self.vocab.id_to_label)
        return score
