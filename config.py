# config file for training or other process
# check the config file before your process
# and import this file to get parameter

# data config
samp_rate = 16000
frame_duration = 0.032
frame_size = int(samp_rate * frame_duration)
feat_dim = frame_size // 2 + 1
shift_duration = 0.008
shift = int(samp_rate * shift_duration)
overlap_rate = 0
batch_size = 32
dev_batch_size = 8
min_queue_size = 64
load_file_num = 64
min_sent_len = 10
shorter_sent_len = 200
longer_sent_len = 500


# data_path
train_noisy_list = './list/train_noisy_wav.lst'
train_clean_list = './list/train_clean_wav.lst'
dev_noisy_list = './list/test_noisy_wav.lst'
dev_clean_list = './list/test_clean_wav.lst'

# job config
job_type = "train"
job_dir =  "job/Conv_Blstm_trial"
gpu_list = [2]

# model config
log_feat = True
global_cmvn_file = ''
conv_channels = [256]
conv_kernels = [[11, 32]]  # each item stand for the kernel for eavh conv layer
                           # kernel shape is [time, frequency]
conv_stride = [[1, 16]]    # strides for batch and channel are 1 by default
num_rnn_layers = 2
rnn_type = 'lstm'  # support 'lstm', 'gru'
hidden_size = 512
bidirectional = True
pred_mask = False

# training param
seed = 123
resume = False
init_mean = 0.0
init_stddev = 0.02
max_grad_norm = 200
learning_rate = 1e-3
max_epoch = 200
pretrain_shorter_epoch = 10
log_period = 10
save_period = 500
dev_period = 500
early_stop_count = 10
decay_lr_count = 3
decay_lr = 0.5
min_learning_rate = 1e-6

# test_config
test_noisy_list = ''
test_clean_list = ''
test_name = ''

# load option 
load_option = 1
load_path = ''
