import csv
import tensorflow as tf
import preprocess
from BIGRU import BiGRU
from ATTENTION import BiAttention, SelfAttention
from FUSION import NonLinearFusion
from CLASSIFICATION import BinaryClassificaion
from METRIC import mertic

#Data path
QA_train_path = "data/QA/QA_train.csv"
QA_dev_path = "data/QA/QA_dev.csv"

QR_train_path = "data/QR/QR_train.csv"
QR_dev_path = "data/QR/QR_dev.csv"

embedding_path = "glove.twitter.27B.50d.txt"
word_dict_path = "word_dict.txt"
log_dir = "logs"

#Model parameters
word_embedding_dim = 50 #set according to embedding_path
max_sequence_length = 80
learning_rate = 0.001
rnn_dim = 32
k = 32
batch_size = 128
epoch = 2  #set 2-->transfer training
threshold = 0.5  # obtain a binary result from prediction result(continuous values), default=0.5
Mu = 0.1
Eta = 0.1
main_iter = 1000  # QR iterations-->Main task iteration


#Prepare Glove and Data
embedding_index, word_id = preprocess.read_word_embedding(embedding_path, word_embedding_dim, word_dict_path)
embedding_matrix = preprocess.pretrained_embedding_matrix(word_id, embedding_index, word_embedding_dim)
vocab_size = len(word_id) + 1

#Read QA training set
QA_train = preprocess.generate_train_test(QA_train_path, word_id, "QA")
QA_q_train = QA_train[0]
QA_a_train = QA_train[1]
QA_y_train = QA_train[2]

#Read QR training set and dev set
QR_train = preprocess.generate_train_test(QR_train_path, word_id, "QR")
QR_q_train = QR_train[0]
QR_r_train = QR_train[1]
QR_y_train = QR_train[2]

QR_dev = preprocess.generate_train_test(QR_dev_path, word_id, "QR")
QR_q_dev = QR_dev[0]
QR_r_dev = QR_dev[1]
QR_y_dev = QR_dev[2]


########Model#########
input_length = tf.fill([batch_size], max_sequence_length)
Q_ori = tf.placeholder(tf.int32, shape=[batch_size, max_sequence_length], name="question_ids")
A_ori = tf.placeholder(tf.int32, shape=[batch_size, max_sequence_length], name="answer_ids")
R_ori = tf.placeholder(tf.int32, shape=[batch_size, max_sequence_length], name="review_ids")
labels = tf.placeholder(tf.float32, shape=[batch_size, 1], name="review_ids")

#Embedding
with tf.variable_scope("Embedding"):
    W = tf.Variable(tf.to_float(embedding_matrix), trainable=False, name="W_Glove")
    Q_embedding = tf.nn.embedding_lookup(W, Q_ori, name="Q_embedding_lookup")
    A_embedding = tf.nn.embedding_lookup(W, A_ori, name="A_embedding_lookup")
    R_embedding = tf.nn.embedding_lookup(W, R_ori, name="R_embedding_lookup")


#BiGRU
Q_gru = BiGRU(Q_embedding, rnn_dim, input_length, batch_size, "Q_GRU")
A_gru = BiGRU(A_embedding, rnn_dim, input_length, batch_size, "A_GRU")
R_gru = BiGRU(R_embedding, rnn_dim, input_length, batch_size, "R_GRU")



#Bi-Attention
with tf.variable_scope("BiAttention"):
    Q_att, A_att, r_Q, r_A = BiAttention(Q_gru, A_gru, rnn_dim, batch_size)

#Self-Attention
with tf.variable_scope("SelfAttention"):
    R_att, r_R = SelfAttention(R_gru, rnn_dim, batch_size, max_sequence_length, k)

#Fusion
with tf.variable_scope("Fusion"):
    uQA = NonLinearFusion(r_Q, r_A, rnn_dim, batch_size)
    gQR = NonLinearFusion(r_Q, r_R, rnn_dim, batch_size)


#Aux-Task (QA-subnet)
with tf.variable_scope("AuxTask"):
    loss_QA, predict_QA = BinaryClassificaion(uQA, labels, rnn_dim, batch_size)
    aux_loss = loss_QA



#Main-Task (QAR-net)
with tf.variable_scope("MainTask"):
    loss_QR, predict_QR = BinaryClassificaion(gQR, labels, rnn_dim, batch_size)
    norm_1 = tf.norm(tf.matmul(Q_att, tf.transpose(R_att, [1, 0])), ord=2)
    norm_2 = tf.norm([r_Q - r_R], ord="euclidean")
    main_loss = loss_QR + Mu * norm_1 + Eta * norm_2
    tf.summary.histogram('predictions', predict_QR)
    tf.summary.scalar('cost', main_loss)


aux_op = tf.train.AdamOptimizer().minimize(aux_loss)
main_op = tf.train.AdamOptimizer().minimize(main_loss)

saver = tf.train.Saver()
init = tf.initialize_all_variables()

csvfile = open("test00.csv", "w", newline='')
writer = csv.writer(csvfile)
writer.writerow(["epoch", "precision", "recall", "f1"])
merged = tf.summary.merge_all()


# Transfer training
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(init)
    step = 0
    for epo in range(epoch):
        # Train
        if epo % 2 == 0:
            print("#########QA training: begin!#########")
            Q_batch, RA_batch, Y_batch = preprocess.batch_pos_neg(QA_q_train, QA_a_train, QA_y_train, batch_size, 0.5, True)
            for i in range(len(Y_batch)):
                _QA, Loss_aux = sess.run([aux_op, aux_loss],
                                     {Q_ori: Q_batch[i], A_ori: RA_batch[i], R_ori: RA_batch[i], labels: Y_batch[i]})
                print("epoch:", epo + 1, "--iter:", i+1, "/", len(Y_batch), "--Loss:", Loss_aux)
        else:
            print("#########QR training: begin!#########")
            for m in range(main_iter):
                Q_batch, RA_batch, Y_batch = preprocess.batch_pos_neg(QR_q_train, QR_r_train, QR_y_train, batch_size, 0.5, True)
                for i in range(len(Y_batch)):
                    _QR, Loss_main, summary = sess.run([main_op, main_loss, merged],
                                            {Q_ori: Q_batch[i], A_ori: RA_batch[i], R_ori: RA_batch[i], labels: Y_batch[i]})
                    train_writer.add_summary(summary, step)
                    print("epoch:", epo + 1, "--round: ", m, "/%d" % (main_iter), "--iter:", i + 1, "/", len(Y_batch), "--loss:", Loss_main)
                    step += 1
        # VALID
                Q_dev_batch, R_dev_batch, Y_dev_batch, last_start = preprocess.batch_triplet_shuffle(QR_q_dev, QR_r_dev, QR_y_dev, batch_size, False)
                prediction = []
                for s in range(len(Y_dev_batch)):
                    Loss_dev, pred = sess.run([main_loss, predict_QR], {Q_ori: Q_dev_batch[s], A_ori: R_dev_batch[s], R_ori: R_dev_batch[s], labels: Y_dev_batch[s]})
                    pred = [pre[0] for pre in pred]
                    if s == len(Y_dev_batch) - 1:
                        prediction += pred[last_start: batch_size]
                    else:
                        prediction += pred
                y_dev_true = [tru[0] for tru in QR_y_dev]
                precision, recall, f1 = mertic(prediction, y_dev_true, threshold)
                print("[epoch_valid_%d]:" % (epo + 1), "--precision: ", precision, "--recall: ", recall, "--f1: ", f1)
                saver.save(sess, "model/model.ckpt"+"_p-%f_r-%f_f1-%f" % (precision, recall, f1))
                writer.writerow([(m + 1), precision, recall, f1] + prediction)
                csvfile.flush()
csvfile.close()






