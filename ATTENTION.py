import tensorflow as tf


def BiAttention(Q_gru, A_gru, rnn_dim, batch_size):
    U = tf.get_variable("U", [2 * rnn_dim, 2 * rnn_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
    G = tf.matmul(tf.matmul(Q_gru, tf.tile(tf.expand_dims(U, 0), [batch_size, 1, 1])), A_gru, adjoint_b=True)
    Q_att = tf.nn.softmax(tf.reduce_max(G, 2))
    A_att = tf.nn.softmax(tf.reduce_max(G, 1))
    r_Q = tf.matmul(tf.transpose(Q_gru, [0, 2, 1]), tf.expand_dims(Q_att, 2))
    r_A = tf.matmul(tf.transpose(A_gru, [0, 2, 1]), tf.expand_dims(A_att, 2))
    return Q_att, A_att, r_Q, r_A


def SelfAttention(R_gru, rnn_dim, batch_size, max_sequence_length, k):
    U_r = tf.get_variable("U_r", [2 * rnn_dim, k], initializer=tf.truncated_normal_initializer(stddev=0.1))
    V_r = tf.get_variable("V_r", [k, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
    U_r_expand = tf.tile(tf.expand_dims(U_r, 0), [batch_size, 1, 1])
    V_r_expand = tf.tile(tf.expand_dims(V_r, 0), [batch_size, 1, 1])
    UH = tf.tanh(tf.matmul(R_gru, U_r_expand))
    R_att = tf.nn.softmax(tf.reshape(tf.matmul(UH, V_r_expand), [batch_size, max_sequence_length]), axis=1)
    r_R = tf.matmul(tf.transpose(R_gru, [0, 2, 1]), tf.expand_dims(R_att, 2))
    return R_att, r_R