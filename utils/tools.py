import os
import sys
proj_path = os.path.abspath('..')
sys.path.append(proj_path)
import logging
import librosa
import subprocess
import re
import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy.io import wavfile
from tensorflow.python.training.moving_averages import assign_moving_average


def create_folders(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def set_log(log_dir):
    log_path = os.path.join(log_dir, 'log.out')
    create_folders(log_dir)
    log_format = ("%(levelname)s %(asctime)s %(filename)s"
                  "[line %(lineno)d] %(message)s")
    logging.basicConfig(level=logging.DEBUG,
                        format=log_format,
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_path,
                        filemode='w')


def read_raw_pcm(path, channels=1, samplerate=16000, subtype='PCM_16'):
    return sf.read(path, channels=channels, samplerate=samplerate, subtype=subtype)


def read_wav(path, offset=0.0, duration=None, samp_rate=16000):
    signal, sr = librosa.load(path, mono=False, sr=samp_rate,
                              offset=offset, duration=duration)
    return signal.astype(np.float32)


def audiowrite(path, data, samp_rate=16000):
    amp_max = max(np.abs(data))
    if amp_max > 1:
        data = data / amp_max
    data = (data + 1) / 2 * 65535 - 32768
    data = data.astype(np.int16)
    wavfile.write(path, samp_rate, data)


def get_file_line(file_path):
    line_list = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()[0]
            line_list.append(line)
    return line_list


def get_SISNR(ref_sig, out_sig, eps=1e-8):
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr


def getPESQ(refWav, tarWav, samp_rate):
    PESQ_path = '/home/fsl/workspace/SpeechSeparation/speech_tools/PESQ'
    fullcmd = os.path.normpath(PESQ_path) + " +" + str(samp_rate) + " " + refWav + " " + tarWav
    pesq_proc = subprocess.Popen(fullcmd, shell=True, stdout=subprocess.PIPE,
                                 universal_newlines=True)
    pesq_out = pesq_proc.communicate()

    # Parse output
    mo_pesq_out = re.compile("Prediction[^=]+=\s+([\-\d\.]+)\s*").search(pesq_out[0])
    if mo_pesq_out == None:
        print("Failed to fetch PESQ result")
        print(fullcmd)
        return -1
    return mo_pesq_out.group(1)


class MetricChecker(object):
    def __init__(self, cfg, less=True):
        self.learning_rate = cfg.learning_rate
        self.min_learning_rate = cfg.min_learning_rate
        self.decay_lr = cfg.decay_lr
        self.decay_lr_count = cfg.decay_lr_count
        self.early_stop_count = cfg.early_stop_count
        self.reset_step()
        self.cur_dev = tf.placeholder(tf.float32, shape=[], name='cur_dev')
        if not less:
            self.best_dev = tf.get_variable(name='best_dev', trainable=False, shape=[],
                                            initializer=tf.constant_initializer(-np.inf))
            self.dev_improved = tf.less(self.best_dev, self.cur_dev)
        else:
            self.best_dev = tf.get_variable(name='best_dev', trainable=False, shape=[],
                                            initializer=tf.constant_initializer(np.inf))
            self.dev_improved = tf.less(self.cur_dev, self.best_dev)
        with tf.control_dependencies([self.dev_improved]):
            if not less:
                self.update_best_dev = tf.assign(self.best_dev,
                                                 tf.maximum(self.cur_dev, self.best_dev))
            else:
                self.update_best_dev = tf.assign(self.best_dev,
                                                 tf.minimum(self.cur_dev, self.best_dev))

    def reset_step(self):
        self.stop_step = 0
        self.lr_step = 0

    def update(self, sess, cur_dev):
        dev_improved, best_dev = sess.run([self.dev_improved, self.update_best_dev],
                                          feed_dict={self.cur_dev: cur_dev})
        if dev_improved:
            self.reset_step()
        else:
            self.stop_step += 1
            self.lr_step += 1
            if self.lr_step == self.decay_lr_count:
                self.lr_step = 0
                self.learning_rate = max(self.learning_rate * self.decay_lr, self.min_learning_rate)
        return dev_improved, best_dev

    def should_stop(self):
        return self.stop_step >= self.early_stop_count

    def get_best(self, sess):
        return sess.run(self.best_dev)


def checker(cfg):
    with tf.variable_scope("DevLoss"):
        snr_checker = MetricChecker(cfg, less=True)
    return snr_checker


def create_valid_summary(dev_loss):
    values = [
        tf.Summary.Value(tag='dev_loss', simple_value=dev_loss)
    ]
    summary = tf.Summary(value=values)
    return summary

def zeros_init():
    return tf.zeros_initializer()

def ones_init():
    return tf.ones_initializer()

def batch_norm(param, masked_param, shape, axes, offset=True, scale=True, eps=1e-6,
               decay=0.999, dtype=tf.float32, scope=None, is_train=True):
    with tf.variable_scope(scope or "BatchNorm"):
        name = param.op.name.split('/')[-1]
        running_mean = tf.get_variable('{}_running_mean'.format(name),
                                       shape=shape, initializer=zeros_init(),
                                       trainable=False, dtype=dtype)
        running_var = tf.get_variable('{}_running_var'.format(name),
                                      shape=shape, initializer=ones_init(),
                                      trainable=False, dtype=dtype)
        offset_var = None
        if offset:
            offset_var = tf.get_variable('{}_offset'.format(name),
                                         shape=shape, initializer=zeros_init(),
                                         dtype=dtype)
        scale_var = None
        if scale:
            scale_var = tf.get_variable('{}_scale'.format(name),
                                        shape=shape, initializer=ones_init(),
                                        dtype=dtype)
        def batch_statistics():
            mean = tf.reduce_mean(masked_param, axes, keep_dims=True)
            var = tf.reduce_mean(tf.square(masked_param - mean), axes, keep_dims=True)
            update_running_mean = assign_moving_average(
                running_mean, mean, decay, zero_debias=False)
            update_running_var = assign_moving_average(
                running_var, var, decay, zero_debias=False)
            with tf.control_dependencies([update_running_mean, update_running_var]):
                normed_param = tf.nn.batch_normalization(
                    param, mean, var, offset_var, scale_var, eps,
                    '{}_bn'.format(name))
            return normed_param
        def population_statistics():
            normed_param = tf.nn.batch_normalization(
                param, running_mean, running_var, offset_var, scale_var,
                eps, '{}_bn'.format(name))
            return normed_param
        normed_param = tf.cond(is_train, batch_statistics, population_statistics)
        return normed_param, running_mean, running_var


def layer_normalization(param, dims, axis=-1, offset=True, scale=True,
                        name=None, eps=1e-6, dtype=tf.float32, scope=None):
    with tf.variable_scope(scope or "LayerNorm"):
        if name is None:
            name = param.op.name.split('/')[-1]
        offset_var = 0
        if offset:
            offset_var = tf.get_variable(name+'_offset', shape=[dims],
                                         initializer=tf.zeros_initializer(),
                                         dtype=dtype)
        scale_var = 1
        if scale:
            scale_var = tf.get_variable(name+'_scale', shape=[dims],
                                        initializer=tf.ones_initializer(),
                                        dtype=dtype)
        mean = tf.reduce_mean(param, axis=axis, keep_dims=True)
        inverse_stddev = tf.rsqrt(tf.reduce_mean(
            tf.square(param - mean), axis=axis, keep_dims=True) + eps)
        normed = (param - mean) * inverse_stddev
        return normed * scale_var + offset_var


def average_gradients(tower_grads, clip_grad):
    average_grads = []
    for grad_and_vars  in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expand_g = tf.expand_dims(g, axis=0)
            grads.append(expand_g)
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, axis=0)
        grad = tf.clip_by_norm(grad, clip_grad)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
