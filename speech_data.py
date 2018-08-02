import os
import sys
import logging
import traceback
import random
import time
import threading
import numpy as np
import config as cfg
try:
    from Queue import Queue
except ImportError:
    from queue import Queue
from utils.signalprocess import *
from utils.tools import *


class Producer(threading.Thread):
    def __init__(self, reader):
        threading.Thread.__init__(self)
        self.reader = reader
        self.exitcode = 0
        self.stop_flag = False

    def run(self):
        try:
            min_queue_size = self.reader._config.min_queue_size
            while not self.stop_flag:
                idx = self.reader._next_load_idx
                if idx >= len(self.reader.data_list):
                    self.reader._batch_queue.put([])
                    break
                if self.reader._batch_queue.qsize() < min_queue_size:
                    batch_list = self.reader.load_samples()
                    for batch in batch_list:
                        self.reader._batch_queue.put(batch)
                else:
                    time.sleep(1)
        except Exception as e:
            logging.warning("producer exception: %s" % e)
            self.exitcode = 1
            traceback.print_exc()

    def stop(self):
        self.stop_flag = True

class SpeechReader(object):
    def __init__(self, config, data_list, batch_size=None, max_sent_len=-1,
                 min_sent_len=10, num_gpu=1, job_type='train'):
        self.num_gpu = num_gpu
        self.max_sent_len = max_sent_len
        self.min_sent_len = min_sent_len
        self.batch_size = batch_size
        if batch_size is None:
            self.batch_size = config.batch_size
        self.eps = 1e-8
        self._config = config
        self.data_list = self.read_data_list(data_list)
        self._job_type = job_type
        self._batch_queue = Queue()
        self.reset()

    def reset(self):
        self.sample_buffer = []
        self._next_load_idx = 0
        if self._job_type == "train":
            self.shuffle_data_list()
        self._producer = Producer(self)
        self._producer.start()

    def shuffle_data_list(self):
        random.shuffle(self.data_list)

    def get_file_line(self, file_path):
        line_list = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.strip().split()[0]
                line_list.append(line)
        return line_list

    def read_data_list(self, data_list):
        noisy_file, clean_file = data_list
        noisy_list = self.get_file_line(noisy_file)
        clean_list = self.get_file_line(clean_file)
        tuple_list = list(zip(noisy_list, clean_list))
        return tuple_list
 
    def load_one_mixture(self, tuple_file):
        sample_list = []
        noisy_file, clean_file = tuple_file
        samp_rate = self._config.samp_rate
        frame_size = self._config.frame_size
        shift = self._config.shift
        noisy_sig = read_wav(noisy_file, samp_rate=samp_rate)
        clean_sig = read_wav(clean_file, samp_rate=samp_rate)
        assert len(noisy_sig) == len(clean_sig)
        sig_len = len(clean_sig)
        seq_len = samples_to_stft_frames(sig_len, frame_size, shift, ceil=False)
        if seq_len < self.min_sent_len:
            return []
        noisy_stft = stft(noisy_sig, size=frame_size, shift=shift,
                          fading=False, ceil=False)
        clean_stft = stft(clean_sig, size=frame_size, shift=shift,
                          fading=False, ceil=False)
        noisy_magn = np.abs(noisy_stft)
        clean_magn = np.abs(clean_stft)
        i = 0
        while self.max_sent_len > 0 and i + self.max_sent_len <= seq_len:
            one_sample = (noisy_magn[i:i+self.max_sent_len],
                          clean_magn[i:i+self.max_sent_len],
                          self.max_sent_len)
            sample_list.append(one_sample)
            i += (1 - self._config.overlap_rate) * self.max_sent_len
        if seq_len - i >= self.min_sent_len and self._job_type != "train":
            one_sample = (noisy_magn[i:], clean_magn[i:], seq_len - i)
            sample_list.append(one_sample)
        return sample_list

    def patch_batch_data(self):
        batch_size = self.batch_size
        group_size = batch_size * self.num_gpu
        feat_dim = self._config.feat_dim
        num_groups = len(self.sample_buffer) // group_size
        if num_groups == 0:
            return []
        group_list = []
        choose_samples = [self.sample_buffer[i:i+group_size]
                          for i in range(0, group_size * num_groups, group_size)]
        self.sample_buffer = self.sample_buffer[group_size * num_groups:]
        for one_group in choose_samples:
            group_magn_noisy = []
            group_magn_clean = []
            group_seq_len = []
            for i in range(0, group_size, batch_size):
                one_batch = one_group[i:i+batch_size]
                max_len =  int(max(map(lambda x: x[2], one_batch)))
                batch_magn_noisy = np.zeros((batch_size, max_len, feat_dim), dtype=np.float32)
                batch_magn_clean = np.zeros((batch_size, max_len, feat_dim), dtype=np.float32)
                batch_seq_len = np.zeros(batch_size, dtype=np.int32)
                for j in range(batch_size):
                    this_len = one_batch[j][2]
                    batch_seq_len[j] = this_len
                    batch_magn_noisy[j, 0:this_len, :] = one_batch[j][0]
                    batch_magn_clean[j, 0:this_len, :] = one_batch[j][1]
                group_magn_noisy.append(batch_magn_noisy)
                group_magn_clean.append(batch_magn_clean)
                group_seq_len.append(batch_seq_len)
            group_list.append((group_magn_noisy, group_magn_clean, group_seq_len))
        return group_list

    def load_samples(self):
        load_file_num = self._config.load_file_num
        idx = self._next_load_idx
        for tuple_file in self.data_list[idx: idx+load_file_num]:
            self.sample_buffer.extend(self.load_one_mixture(tuple_file))
        self._next_load_idx += load_file_num
        if self._job_type == "train":
            random.shuffle(self.sample_buffer)
        group_list = self.patch_batch_data()
        return group_list

    def next_batch(self):
        while self._producer.exitcode == 0:
            try:
                batch_data = self._batch_queue.get(block=False)
                if len(batch_data) == 0:
                    return None
                else:
                    return batch_data
            except Exception as e:
                time.sleep(3)


def test():
    data_list = (cfg.train_noisy_list, cfg.train_clean_list)
    start_time = time.time()
    reader = SpeechReader(cfg, data_list, max_sent_len=200, min_sent_len=10,
                          num_gpu=1, job_type="test")
    batch_data = reader.next_batch()
    magn_noisy, magn_clean, seq_len = batch_data
    print("seg_mix.shape: ", magn_noisy[0].shape, magn_noisy[0].dtype)
    print("seg_src.shape: ", magn_clean[0].shape, magn_clean[0].dtype)
    print("seq_len.shape: ", seq_len[0].shape, seq_len[0].dtype)
    for i in range(99):
        batch_data = reader.next_batch()
    duration = time.time() - start_time
    print("read 100 batches consume {:.2f} seconds".format(duration))
    reader._producer.stop()

if __name__ == "__main__":
    test()
