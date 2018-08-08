import tensorflow as tf


def BinaryClassificaion(u, y,  rnn_dim, batch_size):
    with tf.variable_scope("Classification", reuse=tf.AUTO_REUSE):
        w = tf.get_variable("w", [2 * rnn_dim, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", [batch_size, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
        u = tf.reshape(u, [batch_size, 2 * rnn_dim])
        wu_b = (tf.matmul(u, w) + b)
        predict = tf.nn.sigmoid(wu_b)
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=wu_b, name="binary_classification"))
    return loss, predict