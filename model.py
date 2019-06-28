# -*- coding: utf-8 -*-
"""
Created on Feb 21  10:15  2017

@author: chuito
"""

import os
import time
import numpy as np
import tensorflow as tf

from base import BaseModel
from cell import GRNNSPCell, GRNNSRCell


class Model(BaseModel):

    def __init__(self, sess, reader, embed_dim, hidden_dim, encoder_type="AVG",
                 batch_size=64, dataset="SST", init_embedding=False,
                 learning_rate=0.001, checkpoint_dir="checkpoints", log_dir="logs"):
        self.name = encoder_type

        self._attrs = ["hidden_dim", "embed_dim", "learning_rate"]

        self.sess = sess
        self.reader = reader
        self.dataset = dataset

        self.encoder_type = encoder_type

        self.sequence_length = reader.sequence_length
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.polarity_size = reader.polarity_size

        self.vocab_size = reader.vocab_size
        self.batch_size = batch_size

        self.step = tf.Variable(0, trainable=False)

        self.lr = learning_rate

        self.checkpoint_dir = os.path.join(checkpoint_dir, self.get_model_dir())
        self.log_dir = os.path.join(log_dir, self.get_model_dir())


        self.max_acc = 0
        self.neg_max_acc = 0
        self.inten_max_acc = 0

        # Placeholder
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_l = tf.placeholder(tf.int32, [None], name="input_l")
        self.input_p = tf.placeholder(tf.int32, [None], name="input_p")
        self.input_lex = tf.placeholder(tf.float32, [None, self.sequence_length], name="input_lex")
        self.input_ll = tf.placeholder(tf.float32, [None, self.sequence_length], name="input_ll")
        self.dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="keep_dropout_rate")

        with tf.variable_scope("GRNNSR"):
            # Embedding Layer
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                if init_embedding:
                    init_embed = self.reader.load_initial_embed()
                else:
                    init_embed = np.random.uniform(-0.01, 0.01, size=[self.vocab_size, self.embed_dim]).astype(np.float32)

                self.word_embedding = tf.Variable(init_embed, name="word_embedding", trainable=True)

                # self.source = tf.nn.embedding_lookup(self.word_embedding, self.input_x)
                self.source = tf.nn.dropout(tf.nn.embedding_lookup(self.word_embedding, self.input_x), keep_prob=self.dropout_keep_prob)

                self.mask = tf.sequence_mask(self.input_l, self.sequence_length, dtype=tf.float32)


            # Encoder
            with tf.variable_scope("encoder"):
                if encoder_type == "SUM":
                    sent = tf.reshape(tf.batch_matmul(tf.expand_dims(self.mask, 1), self.source), [-1, self.embed_dim])

                elif encoder_type == "AVG":
                    mask = self.mask / tf.tile(tf.expand_dims(tf.cast(self.input_l, tf.float32), 1),
                                               [1, self.sequence_length])
                    sent = tf.reshape(tf.batch_matmul(tf.expand_dims(mask, 1), self.source), [-1, self.embed_dim])

                elif encoder_type == "CNN":
                    source_expand = tf.expand_dims(self.source, -1)
                    filter_sizes = [3, 4, 5]
                    num_filters = self.hidden_dim // len(filter_sizes)
                    # Create a convolution + maxpool layer for each filter size
                    pooled_outputs = []
                    for i, filter_size in enumerate(filter_sizes):
                        with tf.name_scope("conv-maxpool-%s" % filter_size):
                            # Convolution Layer
                            filter_shape = [filter_size, self.embed_dim, 1, num_filters]
                            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                            b = tf.Variable(tf.constant(0.1, shape=[num_filters], name="b"))
                            conv = tf.nn.conv2d(
                                source_expand,
                                W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv"
                            )
                            # Apply nonlinearity
                            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                            # Maxpooling over the outputs
                            pooled = tf.nn.max_pool(
                                h,
                                ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="pool"
                            )
                            pooled_outputs.append(pooled)

                    # Combine all the pooled features
                    num_filters_total = num_filters * len(filter_sizes)
                    h_pool = tf.concat(3, pooled_outputs)
                    sent = tf.reshape(h_pool, [-1, num_filters_total])

                elif encoder_type == "BiLSTM":
                    fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim // 2, state_is_tuple=True)
                    bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim // 2, state_is_tuple=True)
                    (fw_outputs, bw_outputs), (fw_last_state, bw_last_state) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=fw_cell,
                        cell_bw=bw_cell,
                        inputs=self.source,
                        sequence_length=self.input_l,
                        dtype=tf.float32
                    )
                    sent = tf.concat(1, [fw_last_state[1], bw_last_state[1]])
                    outputs = tf.concat(2, [fw_outputs, bw_outputs])

                else:
                    if encoder_type == "GRU":
                        enc_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim)
                    elif encoder_type == "LSTM":
                        enc_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim, state_is_tuple=True)
                    elif encoder_type == "GRNNSR":
                        enc_cell = GRNNSRCell(num_units=hidden_dim, state_is_tuple=True)
                    elif encoder_type == "GRNNSP":
                        enc_cell = GRNNSPCell(num_units=hidden_dim, state_is_tuple=True)
                    else:
                        raise Exception("Unsuported encoder_type: {}".format(encoder_type))

                    outputs, last_state = tf.nn.dynamic_rnn(
                        cell=enc_cell,
                        inputs=self.source,
                        sequence_length=self.input_l,
                        dtype=tf.float32,
                    )

                    if encoder_type == "GRU":
                        sent = last_state
                    else:
                        sent = last_state[1]

                # self.sent_vec = tf.nn.dropout(sent, keep_prob=self.dropout_keep_prob)
                self.sent_vec = sent

            with tf.variable_scope("output"):
                W = tf.get_variable(
                    name="W_output",
                    shape=[hidden_dim, self.polarity_size],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                b = tf.Variable(tf.constant(0.0, shape=[self.polarity_size], name="b_output"))

                self.scores = tf.nn.xw_plus_b(self.sent_vec, W, b, name="scores")
                self.pred = tf.cast(tf.argmax(self.scores, 1), tf.int32)

                # Sequence test results
                self.sequence_scores = tf.nn.xw_plus_b(tf.reshape(outputs, [-1, hidden_dim]), W, b,
                                                       name="sequence_scores")
                self.word_pred = tf.reshape(tf.cast(tf.argmax(self.sequence_scores, 1), tf.int32),
                                            [-1, self.sequence_length])
                self.word_pred_prob = tf.reshape(tf.reduce_max(tf.nn.softmax(self.sequence_scores), 1),
                                                 [-1, self.sequence_length])

                # Losses
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.scores, self.input_p)
                self.loss = tf.reduce_mean(losses)
                _ = tf.summary.scalar("loss", self.loss)

                # Accuracy
                with tf.variable_scope("accuracy"):
                    self.correct_pred = tf.cast(tf.equal(self.pred, self.input_p), tf.float32)
                    self.accuracy = tf.reduce_mean(self.correct_pred, name="accuracy")
                    _ = tf.summary.scalar("accuracy", self.accuracy)

            with tf.variable_scope("optimize"):
                optimizer = tf.train.AdamOptimizer(self.lr)

                self.optimize_op = optimizer.minimize(self.loss)

                # # Train with gradient clipping
                # tvars = tf.trainable_variables()
                # grads = tf.gradients(self.loss, tvars)
                # for g in grads:
                #     tf.summary.histogram(g.name, g)
                # grads, _ = tf.clip_by_global_norm(grads, 5.0)
                # self.optimize_op = optimizer.apply_gradients(zip(grads, tvars))


    def train(self, epochs):
        # merged_sum = tf.summary.merge_all()
        # writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        tf.global_variables_initializer().run()

        start_time = time.time()
        step = 0
        valid_accs = []
        test_accs = []
        nega_accs = []
        inten_accs = []
        self.max_acc = 0
        self.valid_max_acc = 0
        self.neg_max_acc = 0
        self.inten_max_acc = 0

        start = time.time()

        for epoch in range(epochs):
            for batch in self.reader.batch_iter(batch_size=self.batch_size, data_type="train", shuffle=True):

                x_batch, l_batch, p_batch = zip(*batch)

                feed_dict = {self.input_x: x_batch, self.input_l: l_batch, self.input_p: p_batch, self.dropout_keep_prob: 0.5}

                eval_tensors = [self.optimize_op, self.loss, self.accuracy]

                # if step % 10 == 0:
                #     eval_tensors += [merged_sum]

                eval_ret = self.sess.run(eval_tensors, feed_dict=feed_dict)
                eval_ret = dict(zip(eval_tensors, eval_ret))

                loss = eval_ret[self.loss]
                acc = eval_ret[self.accuracy]

                # if merged_sum in eval_tensors:
                #     writer.add_summary(eval_ret[merged_sum], global_step=step)
                step += 1
                if step % 10 == 0:
                    print("| Epoch Train: {:2d}".format(epoch),
                          "| Step10: {:3d}".format(step//10),
                          "| Time: {:4d} sec".format(int(time.time() - start_time)),
                          "| Loss: {:2.4f}".format(loss),
                          "| Accuracy: {:.3f}".format(acc))

                    # self.save(self.checkpoint_dir, global_step=step)
                    # continue

                    if step % 50 == 0:
                        acc = self.vaild(epoch, step)
                        valid_accs.append(acc)
                        # if acc >= self.valid_max_acc:
                        # self.valid_max_acc = acc
                        acc = self.test_negation(epoch, step)
                        nega_accs.append(acc)
                        acc = self.test_intensity(epoch, step)
                        inten_accs.append(acc)
                        acc = self.test(epoch, step)
                        test_accs.append(acc)


        print("Total time:", time.time()-start)
        exit(0)

        valid_acc_max = np.max(valid_accs)
        valid_acc_mean_5 = np.average(np.sort(valid_accs)[-5:])
        test_acc_max = np.max(test_accs)
        test_acc_mean_5 = np.average(np.sort(test_accs)[-5:])
        nega_acc_max = np.max(nega_accs)
        nega_acc_mean_5 = np.average(np.sort(nega_accs)[-5:])
        inten_acc_max = np.max(inten_accs)
        inten_acc_mean_5 = np.average(np.sort(inten_accs)[-5:])
        print("[NTes] Max: {:.3f}, Mean-5: {:.3f}".format(nega_acc_max, nega_acc_mean_5))
        print("[ITes] Max: {:.3f}, Mean-5: {:.3f}".format(inten_acc_max, inten_acc_mean_5))
        print("[Vali] Max: {:.3f}, Mean-5: {:.3f}".format(valid_acc_max, valid_acc_mean_5))
        print("[Test] Max: {:.3f}, Mean-5: {:.3f}".format(test_acc_max, test_acc_mean_5))


    def vaild(self, epoch, step):
        # Validation
        x_batch, l_batch, p_batch = self.reader.fetch_data(data_type="valid")

        feed_dict = {self.input_x: x_batch, self.input_l: l_batch, self.input_p: p_batch}

        losses, accuracies = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        loss = np.average(losses)
        accuracy = np.average(accuracies)

        print("| Epoch Valid: {:2d}".format(epoch),
              "| Step10: {:3d}".format(step//10),
              "| Loss: {:2.4f}".format(loss),
              "| Accuracy: {:.3f}".format(accuracy))
        return accuracy

    def test(self, epoch, step):
        x_batch, l_batch, p_batch = self.reader.fetch_data(data_type="test")
        feed_dict = {self.input_x: x_batch, self.input_l: l_batch, self.input_p: p_batch}

        losses, accuracies, preds, word_preds = self.sess.run([self.loss, self.accuracy, self.pred, self.word_pred],
                                                              feed_dict=feed_dict)

        loss = np.average(losses)
        accuracy = np.average(accuracies)

        print("| Epoch Test : {:2d}".format(epoch),
              "| Step10: {:3d}".format(step//10),
              "| Loss: {:2.4f}".format(loss),
              "| Accuracy: {:.3f}".format(accuracy))

        if accuracy >= self.max_acc:
            self.max_acc = accuracy
            self.save(self.checkpoint_dir, step)
            self.write_text("test", "./results/results.txt", preds, accuracy, word_preds)

        return accuracy

    def test_negation(self, epoch, step):
        x_batch, l_batch, p_batch = self.reader.fetch_data(data_type="nega_test")
        feed_dict = {self.input_x: x_batch, self.input_l: l_batch, self.input_p: p_batch}

        losses, accuracies, preds, word_preds, word_preds_prob = self.sess.run([self.loss, self.accuracy, self.pred, self.word_pred, self.word_pred_prob],
                                                                               feed_dict=feed_dict)

        loss = np.average(losses)
        accuracy = np.average(accuracies)

        print("| Epoch NTest: {:2d}".format(epoch),
              "| Step10: {:3d}".format(step//10),
              "| Loss: {:2.4f}".format(loss),
              "| Accuracy: {:.3f}".format(accuracy))

        if accuracy > self.neg_max_acc:
            self.neg_max_acc = accuracy
            self.write_text("nega_test", "./results/negation_results.txt", preds, accuracy, word_preds)
            self.write_text("nega_test", "./results/negation_results_probs.txt", preds, accuracy, word_preds_prob)

        return accuracy

    def test_intensity(self, epoch, step):
        x_batch, l_batch, p_batch = self.reader.fetch_data(data_type="inten_test")
        feed_dict = {self.input_x: x_batch, self.input_l: l_batch, self.input_p: p_batch}

        losses, accuracies, preds, word_preds, word_preds_prob = self.sess.run([self.loss, self.accuracy, self.pred, self.word_pred, self.word_pred_prob],
                                                                               feed_dict=feed_dict)

        loss = np.average(losses)
        accuracy = np.average(accuracies)

        print("| Epoch ITest: {:2d}".format(epoch),
              "| Step10: {:3d}".format(step//10),
              "| Loss: {:2.4f}".format(loss),
              "| Accuracy: {:.3f}".format(accuracy))

        if accuracy > self.inten_max_acc:
            self.inten_max_acc = accuracy
            self.write_text("inten_test", "./results/intensity_results.txt", preds, accuracy, word_preds)
            self.write_text("inten_test", "./results/intensity_results_probs.txt", preds, accuracy, word_preds_prob)

        return accuracy

    def write_text_with_score(self, preds, accuracy, taos, base, bias, scores, lambds):
        test_x, test_l, test_p, text_lex, text_ll = self.reader.fetch_data(data_type="test")
        with open("./results.txt", "w") as f:
            f.write("Accuracy: {:.3f}\n".format(accuracy))
            f.write("=============== Error cases ===============\n")
            for target, pred, sent, l, base_score, bias_score, score, lex, tao, lambd in zip(test_p, preds, test_x, test_l, base, bias, scores, text_lex, taos, lambds):
                if pred != target:
                    f.write("{} True: {} Pred: {} Base: {:.2f} Bias: {:.2f} Score: {:.2f} Lambda: {:.2f}\n".format(
                        0, self.reader.id2pola[target], self.reader.id2pola[pred], base_score, bias_score, score, lambd))
                    words = [self.reader.id2word[i] for i in sent[:l]]
                    lex = lex[:l]
                    tao = tao[:l]
                    strings = ["{} ({:.2f}, {:.2f})".format(w, x, x * t) if x != 0 else w for w, x, t in zip(words, lex, tao)]
                    f.write(" ".join(strings) + "\n")
                    f.write("\n")
            for target, pred, sent, l, base_score, bias_score, score, lex, tao, lambd in zip(test_p, preds, test_x, test_l, base, bias, scores, text_lex, taos, lambds):
                if pred == target:
                    f.write("{} True: {} Pred: {} Base: {:.2f} Bias: {:.2f} Score: {:.2f} Lambda: {:.2f}\n".format(
                        1, self.reader.id2pola[target], self.reader.id2pola[pred], base_score, bias_score, score, lambd))
                    words = [self.reader.id2word[i] for i in sent[:l]]
                    lex = lex[:l]
                    tao = tao[:l]
                    strings = ["{} ({:.2f}, {:.2f})".format(w, x, x * t) if x != 0 else w for w, x, t in zip(words, lex, tao)]
                    f.write(" ".join(strings) + "\n")
                    f.write("\n")

    def write_text(self, data_type, file_name, preds, accuracy, word_preds):
        test_x, test_l, test_p = self.reader.fetch_data(data_type=data_type)
        with open(file_name, "w") as f:
            f.write("Accuracy: {:.3f}\n".format(accuracy))
            f.write("=============== Error cases ===============\n")
            i = 0
            for t, p, s, l, w_ps in zip(test_p, preds, test_x, test_l, word_preds):
                i += 1
                if p != t:
                    f.write("{} True: {} Pred: {}\n".format(0, self.reader.id2pola[t], self.reader.id2pola[p]))
                    f.write(str(i) + ". " + " ".join(["{}({:.2f})".format(self.reader.id2word[w], w_p) for w, w_p in zip(s[:l], w_ps[:l])]) + "\n")
                    f.write("\n")
            f.write("=============== Right cases ===============\n")
            i = 0
            for t, p, s, l, w_ps in zip(test_p, preds, test_x, test_l, word_preds):
                i += 1
                if p == t:
                    f.write("{} True: {} Pred: {}\n".format(1, self.reader.id2pola[t], self.reader.id2pola[p]))
                    f.write(str(i) + ". " + " ".join(["{}({:.2f})".format(self.reader.id2word[w], w_p) for w, w_p in zip(s[:l], w_ps[:l])]) + "\n")
                    f.write("\n")



########### Test ################
from reader import SSTReader, MovieReader
from pprint import pprint

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of Adam optimizer (default: 0.001)")
flags.DEFINE_integer("hidden_dim", 300, "The dimension of hidden layer (default: 300)")
flags.DEFINE_integer("embed_dim", 300, "The dimentsion of word embeddings  (default: 300)")
flags.DEFINE_integer("batch_size", 32, "Batch size (default: 32)")
flags.DEFINE_integer("epochs", 2, "Number of training epochs (default: 2)")
flags.DEFINE_string("dataset", "movie", "The name of dataset from [SST, movie] (default: SST)")
flags.DEFINE_string("encoder_type", "GRU", "The type of encoder from [GRU, LSTM, BiLSTM, GRNNSR, GRNNSP] "
                                           "(defalut: GRU)")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory name to save the checkpoints (default: checkpoints)")
flags.DEFINE_boolean("binary", True, "True for binary classification and False for 5-class classification "
                                     "(default: True)")
FLAGS = flags.FLAGS


def main(_):
    pprint(flags.FLAGS.__flags)

    data_path = "./data/{}".format(FLAGS.dataset)

    if FLAGS.dataset == "SST":
        reader = SSTReader(data_path, update_embedding=False, binary=FLAGS.binary)
    elif FLAGS.dataset == "movie":
        reader = MovieReader(data_path, update_embedding=False)
    else:
        raise ValueError("Unsupported dataset: {:s}".format(FLAGS.dataset))

    with tf.Session() as sess:
        model = Model(sess, reader, embed_dim=FLAGS.embed_dim, hidden_dim=FLAGS.hidden_dim,
                      encoder_type=FLAGS.encoder_type, init_embedding=True, dataset=FLAGS.dataset,
                      batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate,
                      checkpoint_dir=FLAGS.checkpoint_dir)

        model.train(FLAGS.epochs)

        ## Test ###
        # dir_base = "5class/GRNNSR" # The checkpoint directory to be loaded
        # checkpoint_dir = os.path.join(dir_base, model.checkpoint_dir.split("/", 1)[1])
        # print("Checkpoint_dir:", checkpoint_dir)
        # if model.load(checkpoint_dir):
        #     model.test(0, 1)
        #     model.test_negation(0, 1)
        #     model.test_intensity(0, 1)

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        print("Exit program early!")