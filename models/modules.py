import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


class FlipGradientBuilder:
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, lw=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * lw]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()


def naive_birnn(inputs, seq_len, num_units, reuse=tf.AUTO_REUSE, name="naive_birnn"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units)
        (o_fw, o_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, seq_len, dtype=tf.float32)
        return o_fw, o_bw


def char_meta_birnn(inputs, seq_len, start_index, end_index, num_units, num_layers, dim, drop_rate=0.0, training=False,
                    activation=tf.tanh, reuse=tf.AUTO_REUSE, name="char_meta_birnn"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        outputs = inputs
        o_fw, o_bw = None, None
        for layer in range(num_layers):
            o_fw, o_bw = naive_birnn(outputs, seq_len, num_units, reuse=reuse, name="naive_birnn_%d" % layer)
            outputs = tf.layers.dropout(tf.concat([o_fw, o_bw], axis=-1), rate=drop_rate, training=training)
        fw_start, fw_end = tf.gather_nd(o_fw, start_index), tf.gather_nd(o_fw, end_index)
        bw_start, bw_end = tf.gather_nd(o_bw, start_index), tf.gather_nd(o_bw, end_index)
        outputs = tf.concat([fw_start, fw_end, bw_start, bw_end], axis=-1)
        outputs = tf.layers.dense(outputs, units=dim, use_bias=True, activation=activation, reuse=reuse, name="dense")
        outputs = tf.layers.dropout(outputs, rate=drop_rate, training=training)
        return outputs


def char_cnn_hw(inputs, kernel_sizes, filters, dim, hw_layers, padding="VALID", activation=tf.nn.relu, use_bias=True,
                hw_activation=tf.tanh, reuse=tf.AUTO_REUSE, name="char_cnn_hw"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        outputs = []
        for i, (kernel_size, filter_size) in enumerate(zip(kernel_sizes, filters)):
            weight = tf.get_variable("filter_%d" % i, shape=[1, kernel_size, dim, filter_size], dtype=tf.float32)
            bias = tf.get_variable("bias_%d" % i, shape=[filter_size], dtype=tf.float32)
            conv = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding=padding, name="conv_%d" % i)
            conv = tf.nn.bias_add(conv, bias=bias)
            pool = tf.reduce_max(activation(conv), axis=2)
            outputs.append(pool)
        outputs = tf.concat(values=outputs, axis=-1)
        for i in range(hw_layers):
            outputs = highway_layer(outputs, num_unit=sum(filters), activation=hw_activation, use_bias=use_bias,
                                    reuse=reuse, name="highway_%d" % i)
        return outputs


def bi_rnn(inputs, seq_len, training, num_units, drop_rate=0.0, activation=tf.tanh, concat=True, use_peepholes=False,
           reuse=tf.AUTO_REUSE, name="bi_rnn"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units, use_peepholes=use_peepholes, name="forward_lstm_cell")
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units, use_peepholes=use_peepholes, name="backward_lstm_cell")
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, seq_len, dtype=tf.float32)
        if concat:
            outputs = tf.concat(outputs, axis=-1)
            outputs = tf.layers.dropout(outputs, rate=drop_rate, training=training)
            outputs = tf.layers.dense(outputs, units=2 * num_units, use_bias=True, activation=activation, name="dense")
        else:
            output1 = tf.layers.dense(outputs[0], units=num_units, use_bias=True, reuse=reuse, name="forward_dense")
            output1 = tf.layers.dropout(output1, rate=drop_rate, training=training)
            output2 = tf.layers.dense(outputs[1], units=num_units, use_bias=True, reuse=reuse, name="backward_dense")
            output2 = tf.layers.dropout(output2, rate=drop_rate, training=training)
            bias = tf.get_variable(name="bias", shape=[num_units], dtype=tf.float32, trainable=True)
            outputs = activation(tf.nn.bias_add(output1 + output2, bias=bias))
        return outputs


def highway_layer(inputs, num_unit, activation, use_bias=True, reuse=tf.AUTO_REUSE, name="highway"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        trans_gate = tf.layers.dense(inputs, units=num_unit, use_bias=use_bias, activation=tf.sigmoid,
                                     name="trans_gate")
        hidden = tf.layers.dense(inputs, units=num_unit, use_bias=use_bias, activation=activation, name="hidden")
        carry_gate = tf.subtract(1.0, trans_gate, name="carry_gate")
        output = tf.add(tf.multiply(hidden, trans_gate), tf.multiply(inputs, carry_gate), name="output")
        return output


def gate_add(inputs1, inputs2, use_bias=True, reuse=tf.AUTO_REUSE, name="gate_add"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        num_units = inputs2.get_shape().as_list()[-1]
        trans_gate = tf.layers.dense(inputs2, units=num_units, use_bias=use_bias, activation=tf.sigmoid, name="trans")
        carry_gate = tf.subtract(1.0, trans_gate, name="carry")
        output = tf.add(tf.multiply(inputs1, trans_gate), tf.multiply(inputs2, carry_gate), name="output")
        return output


def crf_layer(inputs, labels, seq_len, num_units, reuse=tf.AUTO_REUSE, name="crf"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        transition = tf.get_variable(name="transition", shape=[num_units, num_units], dtype=tf.float32)
        crf_loss, transition = tf.contrib.crf.crf_log_likelihood(inputs, labels, seq_len, transition)
        return transition, tf.reduce_mean(-crf_loss)


def embedding_lookup(tokens, token_size, token_dim, token2vec=None, token_weight=None, tune_emb=True, norm_emb=True,
                     project=False, new_dim=None, adversarial_training=False, reuse=tf.AUTO_REUSE, name="lookup_table"):
    with tf.variable_scope(name, reuse=reuse):
        if token2vec is not None:
            table = tf.Variable(initial_value=token2vec, name="table", dtype=tf.float32, trainable=tune_emb)
            unk = tf.get_variable(name="unk", shape=[1, token_dim], trainable=True, dtype=tf.float32)
            table = tf.concat([unk, table], axis=0)
        else:
            table = tf.get_variable(name="table", shape=[token_size - 1, token_dim], dtype=tf.float32, trainable=True)
        if adversarial_training and norm_emb and token_weight is not None:
            weights = tf.constant(np.load(token_weight)["embeddings"], dtype=tf.float32, name="weight",
                                  shape=[token_size - 1, 1])
            table = emb_normalize(table, weights)
        table = tf.concat([tf.zeros([1, token_dim], dtype=tf.float32), table], axis=0)
        token_emb = tf.nn.embedding_lookup(table, tokens)
        if project:
            new_dim = token_dim if new_dim is None else new_dim
            token_emb = tf.layers.dense(token_emb, units=new_dim, use_bias=True, activation=None, reuse=tf.AUTO_REUSE,
                                        name="token_project")
        return token_emb


def emb_normalize(emb, weights):
    mean = tf.reduce_sum(weights * emb, axis=0, keepdims=True)
    var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.0), axis=0, keepdims=True)
    stddev = tf.sqrt(1e-6 + var)
    return (emb - mean) / stddev


def add_perturbation(emb, loss, epsilon=5.0):
    """Adds gradient to embedding and recomputes classification loss."""
    grad, = tf.gradients(loss, emb, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    grad = tf.stop_gradient(grad)
    alpha = tf.reduce_max(tf.abs(grad), axis=(1, 2), keepdims=True) + 1e-12  # l2 scale
    l2_norm = alpha * tf.sqrt(tf.reduce_sum(tf.pow(grad / alpha, 2), axis=(1, 2), keepdims=True) + 1e-6)
    norm_grad = grad / l2_norm
    perturb = epsilon * norm_grad
    return emb + perturb


def self_attention(inputs, return_alphas=False, project=True, reuse=tf.AUTO_REUSE, name="self_attention"):
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        hidden_size = inputs.shape[-1].value
        if project:
            x = tf.layers.dense(inputs, units=hidden_size, use_bias=False, activation=tf.nn.tanh)
        else:
            x = inputs
        weight = tf.get_variable(name="weight", shape=[hidden_size, 1], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=0.01, seed=1227))
        x = tf.tensordot(x, weight, axes=1)
        alphas = tf.nn.softmax(x, axis=-2)
        output = tf.matmul(tf.transpose(inputs, perm=[0, 2, 1]), alphas)
        output = tf.squeeze(output, axis=-1)
        if return_alphas:
            return output, alphas
        else:
            return output


def focal_loss(logits, labels, seq_len=None, weights=None, alpha=0.25, gamma=2):
    label_shape = logits.shape[-1].value
    if label_shape == 2:
        logits = tf.nn.softmax(logits, axis=1)  # logits = tf.nn.sigmoid(logits)
    else:
        logits = tf.nn.softmax(logits, axis=1)
    if labels.get_shape().ndims < logits.get_shape().ndims:
        labels = tf.one_hot(labels, depth=logits.shape[-1].value, axis=-1)
    labels = tf.cast(labels, dtype=tf.float32)
    zeros = tf.zeros_like(logits, dtype=logits.dtype)
    pos_logits_prob = tf.where(labels > zeros, labels - logits, zeros)
    neg_logits_prob = tf.where(labels > zeros, zeros, logits)
    if label_shape == 2:
        cross_entropy = - alpha * (pos_logits_prob ** gamma) * tf.log(tf.clip_by_value(logits, 1e-8, 1.0)) \
                        - (1 - alpha) * (neg_logits_prob ** gamma) * tf.log(tf.clip_by_value(1.0 - logits, 1e-8, 1.0))
    else:
        cross_entropy = - (pos_logits_prob ** gamma) * tf.log(tf.clip_by_value(logits, 1e-8, 1.0)) \
                        - (neg_logits_prob ** gamma) * tf.log(tf.clip_by_value(1.0 - logits, 1e-8, 1.0))
    if weights is not None:
        if weights.get_shape().ndims < logits.get_shape().ndims:
            weights = tf.expand_dims(weights, axis=-1)
        cross_entropy = cross_entropy * weights
    if seq_len is not None:
        mask = tf.sequence_mask(seq_len, maxlen=tf.reduce_max(seq_len), dtype=tf.float32)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=-1)
        cross_entropy = tf.reduce_sum(cross_entropy * mask) / tf.reduce_sum(mask)
    else:
        cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy


def discriminator(features, labels, num_class, grad_rev_rate=0.7, alpha=0.25, gamma=2, mode=0, reuse=tf.AUTO_REUSE,
                  name="discriminator"):
    if mode not in [0, 1, 2]:
        raise ValueError("Unknown mode!!!!")
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        if mode == 0:
            return None
        else:
            feat = flip_gradient(features, lw=grad_rev_rate)
            outputs = self_attention(feat, project=True, reuse=reuse, name="self_attention")
            logits = tf.layers.dense(outputs, units=num_class, use_bias=True, reuse=reuse, name="discriminator_dense")
            if mode == 1:  # normal discriminator
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
                loss = tf.reduce_mean(loss)
            else:  # GRAD
                loss = focal_loss(logits, labels, alpha=alpha, gamma=gamma)
            return loss


def random_mask(prob, mask_shape):
    rand = tf.random_uniform(mask_shape, dtype=tf.float32)
    ones = tf.ones(mask_shape, dtype=tf.float32)
    zeros = tf.zeros(mask_shape, dtype=tf.float32)
    prob = ones * prob
    return tf.where(rand < prob, ones, zeros)


def create_optimizer(cost, lr, decay_step=10, lr_decay=0.99994, opt_name="adam", grad_clip=5.0, name="optimizer"):
    with tf.variable_scope(name):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(lr, global_step, decay_step, lr_decay)
        if opt_name.lower() == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif opt_name.lower() == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif opt_name.lower() == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif opt_name.lower() == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif opt_name.lower() == "lazyadam":
            optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)
        else:  # default adam optimizer
            if opt_name.lower() != 'adam':
                print('Unsupported optimizing method {}. Using default adam optimizer.'.format(opt_name.lower()))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        if grad_clip is not None and grad_clip > 0:
            grads, vs = zip(*optimizer.compute_gradients(cost))
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            train_op = optimizer.apply_gradients(zip(grads, vs), global_step=global_step)
        else:
            train_op = optimizer.minimize(cost, global_step=global_step)
        return train_op


def viterbi_decode(logits, trans_params, seq_len):
    viterbi_sequences = []
    for logit, lens in zip(logits, seq_len):
        logit = logit[:lens]  # keep only the valid steps
        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
        viterbi_sequences += [viterbi_seq]
    return viterbi_sequences
