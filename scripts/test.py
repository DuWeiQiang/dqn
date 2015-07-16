execfile('scripts/load_net.py')

from collections import OrderedDict
n = 4

# Blobs to save for u10f1 model
u10_to_save = OrderedDict()
u10_to_save['frames'] = [n]
u10_to_save['all_frames'] = np.arange(0,320,32) + n
u10_to_save['conv1'] = np.arange(0,320,32) + n
u10_to_save['conv2'] = np.arange(0,320,32) + n
u10_to_save['conv3'] = np.arange(0,320,32) + n

# Blobs to save for the u1f10 model
u1_to_save = OrderedDict()
u1_to_save['frames'] = [n]
u1_to_save['conv1'] = [n]
u1_to_save['conv2'] = [n]
u1_to_save['conv3'] = [n]

u1f10_obs5 = ('scratch/flickr/u1f10obs5_pong_net.prototxt',
              'scratch/flickr/u1f10obs5_pong_iter_10001442.caffemodel',
              'obs5_screens',
              'u1f10_obs5_xray',
              u1_to_save)
u1f10_obs1 = ('scratch/flickr/u1f10obs5_pong_net.prototxt',
              'scratch/flickr/u1f10obs5_pong_iter_10001442.caffemodel',
              'obs1_screens',
              'u1f10_obs1_xray',
              u1_to_save)
u10f1_obs5 = ('scratch/flickr/u10f1obs5_rand_pong_net.prototxt',
              'scratch/flickr/u10f1obs5_rand_pong_HiScore16_iter_5181164.caffemodel',
              'recurrent_obs5_screens',
              'u10f1_obs5_xray',
              u10_to_save)
u10f1_obs1 = ('scratch/flickr/u10f1obs5_rand_pong_net.prototxt',
              'scratch/flickr/u10f1obs5_rand_pong_HiScore16_iter_5181164.caffemodel',
              'recurrent_obs1_screens',
              'u10f1_obs1_xray',
              u10_to_save)
baseline_obs1 = ('scratch/baseline/f4_pong_net.prototxt',
                 'scratch/baseline/f4_pong_HiScore_19.7_std_1.1_iter_3879952.caffemodel',
                 'obs1_screens',
                 'baseline_obs1',
                 u1_to_save)
baseline_obs5 = ('scratch/baseline/f4_pong_net.prototxt',
                 'scratch/baseline/f4_pong_HiScore_19.7_std_1.1_iter_3879952.caffemodel',
                 'obs5_screens',
                 'baseline_obs5',
                 u1_to_save)

models = [u1f10_obs1]

for mdl in models:
  caffe.set_mode_gpu()
  prototxt, snapshot, image_dir, save_dir, to_save = mdl
  net = caffe.Net(prototxt, snapshot, caffe.TEST)
  PrintNet(net)
  # save_maximizing_patches('conv1','conv1', image_dir, save_dir)
  # save_maximizing_patches('conv2','conv2', image_dir, save_dir + '/conv2')
  # save_topn_maximizing_patches('conv1','conv1', image_dir, save_dir + '/conv1_topn')
  # save_topn_maximizing_patches('conv2','conv2', image_dir, save_dir + '/conv2_topn')
  # save_topn_maximizing_patches('conv3','conv3', image_dir, save_dir + '/conv3_topn')
  # save_topn_maximizing_lstm_patches('lstm1','lstm1', image_dir, save_dir + '/max_acts', input_layer_name='all_frames', filter_nums=[22,36,26,356,363,29,178,43,238,325,326,357,393])
  # save_seq_lstm_activation(681, 178, image_dir, save_dir)
  save_lstm_activations(image_dir, save_dir)
  # xray(net, save_dir, xray_params=False, xray_data=to_save)
