import tensorflow as tf


def BiGRU(x, hidden_size, sequence_length, batch_size, name):
    with tf.variable_scope(name):
        fw_cell = tf.contrib.rnn.GRUCell(hidden_size)
        bw_cell = tf.contrib.rnn.GRUCell(hidden_size)
        initial_state_fw = fw_cell.zero_state(batch_size, dtype=tf.float32)
        initial_state_bw = bw_cell.zero_state(batch_size, dtype=tf.float32)
        (outputs, states) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length, initial_state_fw, initial_state_bw)
        out = tf.concat(outputs, 2)
    return out


def BiGRU_share(x, hidden_size, sequence_length, batch_size):
    fw_cell = tf.contrib.rnn.GRUCell(hidden_size)
    bw_cell = tf.contrib.rnn.GRUCell(hidden_size)
    initial_state_fw = fw_cell.zero_state(batch_size, dtype=tf.float32)
    initial_state_bw = bw_cell.zero_state(batch_size, dtype=tf.float32)
    (outputs, states) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length, initial_state_fw, initial_state_bw)
    out = tf.concat(outputs, 2)
    return out