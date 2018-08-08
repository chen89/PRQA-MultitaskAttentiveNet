import csv
from nltk.tokenize import WordPunctTokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import random
import os

QA_train_path = "data/QA/QA_train.csv"
QA_dev_path = "data/QA/QA_dev.csv"

QR_train_path = "data/QR/QR_train.csv"
QR_dev_path = "data/QR/QR_dev.csv"

#Stanford Glove
embedding_path = "glove.6B.50d.txt"
word_dict_path = "word_dict.txt"
word_embedding_dim = 50
MAX_SEQUENCE_LENGTH = 80


def sentence_to_words(sent):
    words = WordPunctTokenizer().tokenize(sent)
    return words

#This fuction is time consuming
#Return embedding_index: {id: embedding} and word_id: {word: id}
def read_word_embedding(embedding_path, word_embedding_dim, word_dict_path):

    embeddings_index = {}
    word_id = {}
    f = open(embedding_path, "r", encoding="utf-8")
    word_dict = open(word_dict_path, "w", encoding="utf-8")
    ids = 1
    #OOV words init by zeros
    embeddings_index[0] = np.zeros(word_embedding_dim, dtype="float32")
    for line in f:
        values = line.split()
        #filter the strange word sequences which contains more than 1 blank
        if len(values) == word_embedding_dim + 1:
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[ids] = coefs
            word_id[word] = ids
            word_dict.write(word+" "+str(ids)+"\n")
            ids += 1
        else:
            continue
    f.close()
    word_dict.close()
    return embeddings_index, word_id


#word sequences to id sequences
def word_to_ids(word_sequence, word_id):
    words_id_sequence = []
    for word in word_sequence:
        if (word not in word_id) and (word.lower() not in word_id):
            words_id_sequence.append(0)
        elif (word not in word_id) and (word.lower() in word_id):
            words_id_sequence.append(word_id[word.lower()])
        else:
            words_id_sequence.append(word_id[word])
    return words_id_sequence


#Read QA data
#Return words/id triples: (question, answer, label)
def read_QA(QA_path, word_id):
    QA_words = []
    QA_ids = []
    data_csv = csv.reader(open(QA_path, "r"))
    for item in data_csv:
        question = sentence_to_words(item[1].strip())
        answer = sentence_to_words(item[2].strip())
        label = int(item[3].strip())
        question_ids = word_to_ids(question, word_id)
        answer_ids = word_to_ids(answer, word_id)
        QA_words.append((question, answer, label))
        QA_ids.append((question_ids, answer_ids, label))
    return QA_words, QA_ids


#Read QR data
#Return words/id triples: (question, review, label)
def read_QR(QR_path, word_id):
    QR_words = []
    QR_ids = []
    data_csv = csv.reader(open(QR_path, "r"))
    for item in data_csv:
        question = sentence_to_words(item[1].strip())
        review = sentence_to_words(item[2].strip())
        label = int(item[4].strip())
        question_ids = word_to_ids(question, word_id)
        review_ids = word_to_ids(review, word_id)
        QR_words.append((question, review, label))
        QR_ids.append((question_ids, review_ids, label))
    return QR_words, QR_ids


#Init pretrained embedding
def pretrained_embedding_matrix(word_id, embedding_index, word_embedding_dim):
    embedding_matrix = np.zeros((len(word_id) + 1, word_embedding_dim))
    for word, i in word_id.items():
        embedding_vector = embedding_index[i]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def generate_train_test(QR_QA_path, word_id, read_pattern):
    data_ids = None
    if read_pattern == "QR":
        data_ids = read_QR(QR_QA_path, word_id)[1]
        print("Read QR data...")
    elif read_pattern == "QA":
        print("Read QA data, time consuming...")
        data_ids = read_QA(QR_QA_path, word_id)[1]
    else:
        print("Invalid argument: read_pattern should be QA or QR!")
    question = []
    review_answer = []
    label = []
    for triple in data_ids:
        question.append(triple[0])
        review_answer.append(triple[1])
        label.append([triple[2]])
    q = pad_sequences(question, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    r_a = pad_sequences(review_answer, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    y = np.asarray(label)
    return q, r_a, y


def batch_triplet_shuffle(Q, RA, Y, batch_size, shuff=False):
    total_num = len(Y)
    last_start = batch_size - (total_num - int(total_num / batch_size) * batch_size)
    data_index = [num for num in range(total_num)]
    if shuff == True:
        random.shuffle(data_index)
    Q = [Q[idx] for idx in data_index]
    RA = [RA[idx] for idx in data_index]
    Y = [Y[idx] for idx in data_index]
    Q_batch = []
    RA_batch = []
    Y_batch = []
    if batch_size >= total_num:
        Q_batch = Q
        RA_batch = RA
        Y_batch = Y
    else:
        for i in range(0, total_num, batch_size):
            if (i + batch_size) > total_num:
                Q_batch.append(Q[total_num - batch_size: total_num])
                RA_batch.append(RA[total_num - batch_size: total_num])
                Y_batch.append(Y[total_num - batch_size: total_num])
            else:
                Q_batch.append(Q[i: i + batch_size])
                RA_batch.append(RA[i: i + batch_size])
                Y_batch.append(Y[i: i + batch_size])

    return Q_batch, RA_batch, Y_batch, last_start


#ratio = pos/all, set 0.5 means pos:neg=1:1.
# under-sampling from majority class(negative)
def batch_pos_neg(Q, RA, Y, batch_size, ratio=0.5, shuff=False):
    total_num = len(Y)
    data_index = [num for num in range(total_num)]
    if shuff == True:
        random.shuffle(data_index)
    Q = [Q[idx] for idx in data_index]
    RA = [RA[idx] for idx in data_index]
    Y = [Y[idx] for idx in data_index]

    pos_Q = [Q[idx] for idx in data_index if Y[idx] == 1]
    pos_RA = [RA[idx] for idx in data_index if Y[idx] == 1]
    pos_Y = [Y[idx] for idx in data_index if Y[idx] == 1]

    neg_Q = [Q[idx] for idx in data_index if Y[idx] == 0]
    neg_RA = [RA[idx] for idx in data_index if Y[idx] == 0]
    neg_Y = [Y[idx] for idx in data_index if Y[idx] == 0]


    pos_batch_size = int(ratio * batch_size)
    neg_batch_size = batch_size - pos_batch_size


    Q_pos_batch = []
    RA_pos_batch = []
    Y_pos_batch = []

    Q_neg_batch = []
    RA_neg_batch = []
    Y_neg_batch = []

    Q_batch = []
    RA_batch = []
    Y_batch = []


    if batch_size >= total_num:
        Q_batch = Q
        RA_batch = RA
        Y_batch = Y
    else:
        for i in range(0, len(pos_Q), pos_batch_size):
            if i + pos_batch_size > len(pos_Q):
                Q_pos_batch.append(pos_Q[len(pos_Q) - pos_batch_size: len(pos_Q)])
                RA_pos_batch.append(pos_RA[len(pos_Q) - pos_batch_size: len(pos_Q)])
                Y_pos_batch.append(pos_Y[len(pos_Q) - pos_batch_size: len(pos_Q)])
            else:
                Q_pos_batch.append(pos_Q[i: i + pos_batch_size])
                RA_pos_batch.append(pos_RA[i: i + pos_batch_size])
                Y_pos_batch.append(pos_Y[i: i + pos_batch_size])
        # print(len(Y_pos_batch),len(Q_pos_batch))
        # print(Q_pos_batch[0][0], Y_pos_batch[0][0])
        for i in range(0, len(neg_Q), neg_batch_size):
            if i + neg_batch_size > len(neg_Q):
                Q_neg_batch.append(neg_Q[len(neg_Q) - neg_batch_size: len(neg_Q)])
                RA_neg_batch.append(neg_RA[len(neg_Q) - neg_batch_size: len(neg_Q)])
                Y_neg_batch.append(neg_Y[len(neg_Q) - neg_batch_size: len(neg_Q)])
            else:
                Q_neg_batch.append(neg_Q[i: i + neg_batch_size])
                RA_neg_batch.append(neg_RA[i: i + neg_batch_size])
                Y_neg_batch.append(neg_Y[i: i + neg_batch_size])
        # print(len(Y_neg_batch), len(Q_neg_batch))
        # print(Q_neg_batch[0][0], Y_neg_batch[0][0])
        for n in range(0, len(Y_pos_batch)):
            Q_b = Q_pos_batch[n] + Q_neg_batch[n]
            RA_b = RA_pos_batch[n] + RA_neg_batch[n]
            Y_b = Y_pos_batch[n] + Y_neg_batch[n]
            Q_batch.append(Q_b)
            RA_batch.append(RA_b)
            Y_batch.append(Y_b)

    return Q_batch, RA_batch, Y_batch
















