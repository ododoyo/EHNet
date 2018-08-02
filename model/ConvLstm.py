import os
import sys
project_path = os.path.abspath('..')
sys.path.append(project_path)

import logging
import time
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from utils.tools import *


class ConvLstmModel(object):
    def __init__(self, sess, config, num_gpu, initializer=None):
        self.session = sess
        self.config = config
        self.num_gpu = num_gpu
        self.epoch_counter = 0
        self.initializer = initializer
        self.eps = 1e-8
        if hasattr(config, 'global_cmvn_file') and config.global_cmvn_file != '':
            self.cmvn = self.read_cmvn(config.global_cmvn_file)
        else:
            self.cmvn = None
        self.global_step = tf.get_variable("global_step", shape=[], trainable=False,
                                           initializer=tf.constant_initializer(0),
                                           dtype=tf.int32)
        # define placeholder
        self.lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.create_placeholder()
        self.training = tf.placeholder(tf.bool, shape=[])
        # init graph
        self.optimize()
        self.reset()
        # create job_env
        self.job_dir = config.job_dir
        create_folders(self.job_dir)
        self.best_loss_dir = os.path.join(config.job_dir, 'best_loss')
        create_folders(self.best_loss_dir)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        self.best_loss_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        train_event_dir = os.path.join(config.job_dir, 'train_event')
        dev_event_dir = os.path.join(config.job_dir, 'dev_event')
        create_folders(train_event_dir)
        create_folders(dev_event_dir)
        self.train_writer = tf.summary.FileWriter(train_event_dir, sess.graph)
        self.dev_writer = tf.summary.FileWriter(dev_event_dir)

    def read_cmvn(self, file_path):
        cmvn = np.loadtxt(file_path).astype(np.float32)
        cmvn[:, 1] = np.sqrt(cmvn[:, 1])
        return cmvn

    def create_placeholder(self):
        self._input = []
        self._target = []
        self._seq_len = []
        feat_dim = self.config.feat_dim
        for i in range(self.num_gpu):
            self._input.append(tf.placeholder(tf.float32, shape=[None, None, feat_dim]))
            self._target.append(tf.placeholder(tf.float32, shape=[None, None, feat_dim]))
            self._seq_len.append(tf.placeholder(tf.int32, shape=[None]))

    def reset(self):
        self.batch_counter = 0
        self.total_loss = 0
        self.latest_loss = 0
        self.latest_batch_counter = 0
        self.epoch_counter += 1

    def transfrom_input(self, inputs):
        if self.cmvn is not None:
            trans_input = (inputs - self.cmvn[:, 0]) / self.cmvn[:, 1]
        else:
            trans_input = inputs
        trans_input = tf.expand_dims(trans_input, axis=3)
        # shape is [B, T, F, C]
        return trans_input

    def ConvLstmNet(self, inputs, seq_len):
        assert len(inputs.get_shape().as_list()) == 4
        feat_dim = self.config.feat_dim
        batch_size = tf.shape(inputs)[0]
        time_step = tf.shape(inputs)[1]
        conv_input = inputs
        conv_w = feat_dim
        for i in range(len(self.config.conv_channels)):
            with tf.variable_scope("cnn_%d"%(i+1)):
                in_channel = conv_input.get_shape()[-1]
                out_channel = int(self.config.conv_channels[i])
                k_h, k_w = self.config.conv_kernels[i]
                k_h, k_w = int(k_h), int(k_w)
                assert k_h % 2 != 0
                pad_h = (k_h - 1) // 2
                pad = tf.constant([[0,0], [pad_h, pad_h], [0,0], [0,0]])
                conv_input = tf.pad(conv_input, pad, mode='CONSTANT')
                stride = [1,] + self.config.conv_stride[i] + [1,]
                kernel = tf.get_variable("kernel", shape=[k_h, k_w, in_channel, out_channel])
                conv_input = tf.nn.conv2d(conv_input, kernel, stride, padding='VALID')
                conv_input = tf.nn.relu(conv_input)
                conv_w = np.ceil((conv_w - k_w + 1) / stride[2]).astype(np.int32)
        rnn_inputsize = conv_w * out_channel
        rnn_input = tf.reshape(conv_input, [batch_size, time_step, rnn_inputsize])
        # print("rnn_input size: {}".format(rnn_input.get_shape()[2]))
        rnn_type = self.config.rnn_type
        hidden_size = self.config.hidden_size
        if rnn_type == "lstm":
            rnn_cell = tf.contrib.rnn.LSTMCell
        elif rnn_type == "gru":
            rnn_cell = tf.contrib.rnn.GRUCell
        else:
            raise ValueError("Not supported rnn_type: {}".format(rnn_type))
        for i in range(self.config.num_rnn_layers):
            with tf.variable_scope("{}_{}".format(rnn_type, i+1)):
                if self.config.bidirectional:
                    fw_cell = rnn_cell(hidden_size, use_peepholes=True) if rnn_type=='lstm' else rnn_cell(hidden_size)
                    bw_cell = rnn_cell(hidden_size, use_peepholes=True) if rnn_type=='lstm' else rnn_cell(hidden_size)
                    initial_fw = fw_cell.zero_state(batch_size, dtype=tf.float32)
                    initial_bw = bw_cell.zero_state(batch_size, dtype=tf.float32)
                    output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, rnn_input,
                                                                sequence_length=tf.to_int32(seq_len),
                                                                initial_state_fw=initial_fw,
                                                                initial_state_bw=initial_bw,
                                                                time_major=False, dtype=tf.float32)
                    output = tf.concat(output, axis=2)
                    rnn_input = output
                else:
                    fw_cell = rnn_cell(hidden_size, use_peepholes=True) if rnn_type=='lstm' else rnn_cell(hidden_size)
                    initial_fw = fw_cell.zero_state(batch_size, dtype=tf.float32)
                    output, _ = tf.nn.dynamic_rnn(fw_cell, rnn_input,
                                                  sequence_length=tf.to_int32(seq_len),
                                                  initial_state=initial_fw,
                                                  time_major=False, dtype=tf.float32)
                    rnn_input = output
        outdims = hidden_size * 2 if self.config.bidirectional else hidden_size
        reshape_out = tf.reshape(output, [-1, outdims])
        reshape_out = tcl.fully_connected(inputs=reshape_out, num_outputs=feat_dim, activation_fn=tf.nn.relu)
        output = tf.reshape(reshape_out, [batch_size, time_step, feat_dim])
        return output

    def get_padding_mask(self, max_len, seq_len):
        r = tf.range(max_len)
        func = lambda x: tf.cast(tf.less(r, x), tf.float32)
        mask = tf.map_fn(func, seq_len, dtype=tf.float32)
        return mask

    def tower_cost(self, inputs, targets, seq_len):
        if self.config.log_feat:
            trans_input = tf.log(1 + inputs)
        else:
            trans_input = inputs
        trans_input = self.transfrom_input(trans_input)
        max_len = tf.shape(inputs)[1]
        padding_mask = self.get_padding_mask(max_len, seq_len)
        regress_out = self.ConvLstmNet(trans_input, seq_len)
        if self.config.pred_mask:
            pred = regress_out * inputs
        elif self.config.log_feat:
            pred = tf.exp(regress_out) - 1
        else:
            pred = regress_out
        feat_dim = self.config.feat_dim
        count_bins = seq_len * feat_dim
        padding_mask = tf.expand_dims(padding_mask, axis=2)
        masked_pred = pred * padding_mask
        masked_targets = targets * padding_mask
        loss = tf.reduce_sum((masked_pred - masked_targets) ** 2, axis=(1, 2))
        loss = loss / tf.cast(count_bins, tf.float32)
        loss = tf.reduce_mean(loss)
        return loss, pred

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        tower_grads = []
        tower_cost = []
        tower_pred = []
        for i in range(self.num_gpu):
            worker = '/gpu:%d' % i
            device_setter = tf.train.replica_device_setter(
                worker_device=worker, ps_device='/cpu:0', ps_tasks=1)
            with tf.variable_scope("Model", reuse=(i>0)):
                with tf.device(device_setter):
                    with tf.name_scope("tower_%d" % i) as scope:
                        cost, pred = self.tower_cost(self._input[i], self._target[i], self._seq_len[i])
                        grads = optimizer.compute_gradients(cost)
                        tower_grads.append(grads)
                        tower_cost.append(cost)
                        tower_pred.append(pred)
        grads = average_gradients(tower_grads, self.config.max_grad_norm)
        self.apply_gradients_op = optimizer.apply_gradients(grads, global_step=self.global_step)
        self.avg_cost = tf.reduce_mean(tower_cost)
        self.tower_pred = tower_pred
        tf.summary.scalar('avg_cost', self.avg_cost)
        self.merged = tf.summary.merge_all()

    def run_batch(self, group_data, learning_rate):
        feed_dict = {self.training: True, self.lr: learning_rate}
        step_size = 0
        for i in range(self.num_gpu):
            feed_dict[self._input[i]] = group_data[0][i]
            feed_dict[self._target[i]] = group_data[1][i]
            feed_dict[self._seq_len[i]] = group_data[2][i]
            step_size += len(group_data[2][i])
        start_time = time.time()
        _, i_global, i_merge, loss = self.session.run(
            [self.apply_gradients_op, self.global_step, self.merged, self.avg_cost],
            feed_dict=feed_dict)
        self.total_loss += loss
        self.latest_loss += loss
        self.batch_counter += 1
        self.latest_batch_counter += 1
        duration = time.time() - start_time
        if i_global % self.config.log_period == 0:
            logging.info("Epoch {:d}, Average Train MSE: {:.6f}={:.6f}/{:d}, "
                         "Latest MSE: {:.6f}, Speed: {:.2f} sentence/sec".format(
                         self.epoch_counter, self.total_loss / self.batch_counter,
                         self.total_loss, self.batch_counter,
                         self.latest_loss / self.latest_batch_counter,
                         step_size / duration))
            self.latest_batch_counter = 0
            self.latest_loss = 0
            self.train_writer.add_summary(i_merge, i_global)
        if i_global % self.config.save_period == 0:
            self.save_model(i_global)
        return i_global

    def get_pred(self, group_data):
        feed_dict = {self.training: False}
        for i in range(self.num_gpu):
            feed_dict[self._input[i]] = group_data[0][i]
            feed_dict[self._target[i]] = group_data[1][i]
            feed_dict[self._seq_len[i]] = group_data[2][i]
        pred = self.session.run(self.tower_pred, feed_dict=feed_dict)
        return pred

    def valid(self, reader):
        total_loss, batch_counter = 0.0, 0
        num_sent = 0
        logging.info("Start to dev")
        start_time = time.time()
        while True:
            batch_data = reader.next_batch()
            if batch_data == None:
                break
            else:
                feed_dict = {self.training: False}
                for i in range(self.num_gpu):
                    feed_dict[self._input[i]] = batch_data[0][i]
                    feed_dict[self._target[i]] = batch_data[1][i]
                    feed_dict[self._seq_len[i]] = batch_data[2][i]
                    num_sent += len(batch_data[2][i])
                loss = self.session.run(self.avg_cost, feed_dict=feed_dict)
                total_loss += loss
                batch_counter += 1
                if batch_counter % 10 == 0:
                    logging.info("Dev Sentence {:d}, AVG Dev MSE: {:.6f}={:.6f}/{:d}, "
                                 "Speed: {:.2f} sentence/sec".format(
                                 num_sent, total_loss / batch_counter, total_loss,
                                 batch_counter, num_sent / (time.time() - start_time)))
        duration = time.time() - start_time
        avg_loss = total_loss / batch_counter
        dev_summary = create_valid_summary(avg_loss)
        i_global = self.session.run(self.global_step)
        self.dev_writer.add_summary(dev_summary, i_global)
        logging.info("Finish dev {:d} sentences in {:.2f} seconds, "
                     "AVG MSE: {:.6f}".format(num_sent, duration, avg_loss))
        return avg_loss

    def save_model(self, i_global):
        model_path = os.path.join(self.job_dir, 'model.ckpt')
        self.saver.save(self.session, model_path, global_step=i_global)
        logging.info("Saved model, global_step={}".format(i_global))

    def restore_model(self):
        load_option = self.config.load_option
        if load_option == 0:
            load_path = tf.train.latest_checkpoint(self.job_dir)
        elif load_option == 1:
            load_path = tf.train.latest_checkpoint(self.best_loss_dir)
        else:
            load_path = self.config.load_path
        try:
            self.saver.restore(self.session, load_path)
            logging.info("Loaded model from path {}".format(load_path))
        except Exception as e:
            logging.error("Failed to load model from {}".format(load_path))
            raise e
