#!/usr/bin/ipython -i
import os
from os.path import join, isfile
import numpy as np
import matplotlib.pyplot as plt
import sys
from random import seed, choice
caffe_root = '/u/mhauskn/projects/recurrent_caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def load_frame_data(fname):
  return np.fromfile(open(fname,'rb'), dtype=np.uint8).reshape(4,84,84).astype(np.float32)

def deprocess(input_, mean=None, input_scale=None,
               raw_scale=None, channel_order=None):
  decaf_in = input_.copy().squeeze()
  if input_scale is not None:
      decaf_in /= input_scale
  if mean is not None:
      decaf_in += mean
  if raw_scale is not None:
      decaf_in /= raw_scale
  decaf_in = decaf_in.transpose((1,2,0))
  if channel_order is not None:
      channel_order_inverse = [channel_order.index(i)
                               for i in range(decaf_in.shape[2])]
      decaf_in = decaf_in[:, :, channel_order_inverse]
  return decaf_in

def run_forward(image_dir, input_layer_name):
  assert os.path.isdir(image_dir)
  assert input_layer_name in net.blobs.keys()
  n,c,w,h = net.blobs[input_layer_name].data.shape
  assert w == 84 and h == 84
  files = [f for f in os.listdir(image_dir) if isfile(join(image_dir,f)) ]
  fnum = 0
  images = []
  def load_frame_data(fname):
    return np.fromfile(open(fname,'rb'), dtype=np.uint8)\
             .reshape(84,84).astype(np.float32)
    # return np.ones([84,84], dtype=np.float32)
  for i in xrange(n):
    frames = []
    for j in xrange(c):
      frames.append(load_frame_data(join(image_dir, files[fnum])))
      fnum += 1
    joined_frames = np.asarray(frames)
    images.append(joined_frames)
  input_frames = np.asarray(images)
  return forward_from_frames(input_frames, input_layer_name)

def forward_from_frames(input_frames, input_layer_name):
  n,c,w,h = net.blobs[input_layer_name].data.shape
  kUnroll = 2
  cont = np.ones([kUnroll,n,1,1], dtype=np.float32)
  targets = np.zeros([kUnroll,n,18,1], dtype=np.float32)
  filters = np.ones([kUnroll,n,18,1], dtype=np.float32)
  net.set_input_arrays(0, input_frames, np.zeros([n,1,1,1], dtype=np.float32))
  net.set_input_arrays(2, cont, np.zeros([kUnroll,1,1,1], dtype=np.float32))
  net.set_input_arrays(3, targets, np.zeros([kUnroll,1,1,1], dtype=np.float32))
  net.set_input_arrays(4, filters, np.zeros([kUnroll,1,1,1], dtype=np.float32))
  net.forward()
  net.backward()
  return input_frames

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(input_data, padsize=1, padval=0, title='', fname=''):
  data = np.copy(input_data)
  data -= data.min()
  data /= data.max()
  # force the number of filters to be square
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = ((0, n ** 2 - data.shape[0]),
             (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
  data = np.pad(data, padding, mode='constant',
                constant_values=(padval, padval))
  # tile the filters into an image
  data = data.reshape(
    (n, n) + data.shape[1:]).transpose(
      (0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
  data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
  plt.clf()
  plt.imshow(data)
  plt.title(title)
  if not fname:
    plt.show()
  else:
    plt.savefig(fname)

# Visualize a single filter sliced along input dimension
def vis_filter(layer_name, filter_num, fname=''):
  all_filters = net.params[layer_name][0].data
  filter = all_filters[filter_num]
  try:
    bias = np.squeeze(net.params[layer_name][1].data)[filter_num]
  except IndexError:
    bias = np.squeeze(net.params[layer_name][1].data).item()
  title = '[Filter] Layer=%s Num=%d (%.3f,%.3f,%.3f) B=%.3f'\
          %(layer_name, filter_num, np.min(filter),
            np.mean(filter), np.max(filter), bias)
  vis_square(filter, title=title, fname=fname)

# Visualize all filters for a given input dimension
def vis_dim(layer_name, input_dim, fname=''):
  filters = net.params[layer_name][0].data[:,input_dim]
  title = '[Filters] Layer=%s InputDim=%d (%.3f,%.3f,%.3f)'\
          %(layer_name, input_dim, np.min(filters),
            np.mean(filters), np.max(filters))
  vis_square(filters, title=title, fname=fname)

# Visualize the mean filters
def vis_mean_filters(layer_name, fname=''):
  filters = net.params[layer_name][0].data
  mean_filters = np.mean(filters, axis=1)
  title = '[MeanFilters] Layer=%s (%.3f,%.3f,%.3f)'\
          %(layer_name, np.min(filters), np.mean(filters), np.max(filters))
  vis_square(mean_filters, title=title, fname=fname)

# Visualize the activations for a given layer
def vis_activations(layer_name, fname='', num=0):
  activations = net.blobs[layer_name].data[num]
  title = '[Activations] Blob=%s Num=%d (%.3f,%.3f,%.3f)'\
          %(layer_name, num, np.min(activations), np.mean(activations),
            np.max(activations))
  vis_square(activations, title=title, fname=fname, padval=1)

# Reshape list specifies how weights should be reshaped
def vis_fc_incoming_weights(layer_name, activation=None, reshape=None, fname='',
                            unit=0, num=0):
  weights = net.params[layer_name][0].data[num,0,unit,:]
  if reshape is not None:
    weights = weights.reshape(reshape)
  title = '[FC Weights] Layer=%s Num=%d Unit=%d Act=%.3f (%.3f,%.3f,%.3f)'\
          %(layer_name, num, unit, activation,
            np.min(weights), np.mean(weights), np.max(weights))
  vis_square(weights, title=title, fname=fname)

def vis_weights(layer_name, fname=''):
  weights = net.params[layer_name][0].data
  while len(weights.shape) < 3:
    weights = np.expand_dims(weights, axis=0)
  title = '[Weights] Layer=%s (%.3f,%.3f,%.3f)'\
          %(layer_name, np.min(weights), np.mean(weights), np.max(weights))
  vis_square(weights, title=title, fname=fname)

def vis_biases(layer_name, fname=''):
  data = net.params[layer_name][1].data
  num = len(data.flatten())
  n = int(np.ceil(np.sqrt(num)))
  viz_data = np.zeros(n**2)
  viz_data[:num] = data
  title = '[Biases] Layer=%s Total=%d (%.3f,%.3f,%.3f)'\
          %(layer_name, num, np.min(data), np.mean(data), np.max(data))
  plt.imshow(viz_data.reshape((n,n)))
  plt.title(title)
  if not fname:
    plt.show()
  else:
    plt.savefig(fname)

def save_blobs(net, save_dir, use_diff=False):
  blob_num = 0
  for blob_name in net.blobs.keys():
    blob_num += 1
    blob = net.blobs[blob_name]
    data = blob.diff if use_diff else blob.data
    shape = data.shape
    squeezed = np.squeeze(data)
    padsize=1
    if len(squeezed.shape) > 2:
      for n in xrange(squeezed.shape[0]):
        d = squeezed[n]
        while len(d.shape) < 3:
          d = np.expand_dims(d, axis=0)
          padsize=0
        title = 'Blob=%s Shape=%s Num=%s (%.3f, %.3f, %.3f)'\
                %(blob_name, shape, n, np.min(d), np.mean(d), np.max(d))
        print '[X-Ray] Saving Blob:', blob_name, 'Num =', n, 'Shape =', shape
        fname = 'blob%i_%s_n%i.png'%(blob_num,blob_name,n)
        vis_square(d, title=title, fname=join(save_dir,fname),
                   padsize=padsize, padval=1)
    else:
      while len(squeezed.shape) < 3:
        squeezed = np.expand_dims(squeezed, axis=0)
        padsize=0
      title = 'Blob=%s Shape=%s (%.3f, %.3f, %.3f)'\
              %(blob_name, shape, np.min(data), np.mean(data), np.max(data))
      fname = 'blob%i_%s.png'%(blob_num,blob_name)
      print '[X-Ray] Saving Blob:', blob_name, 'Shape =', shape
      vis_square(squeezed, title=title, fname=join(save_dir,fname),
                 padsize=padsize, padval=1)

def save_params(net, save_dir, iteration=None, cnt=None):
  iteration = 'Iter=%s'%iteration if iteration is not None else ''
  cnt = '_%02d'%cnt if cnt is not None else ''
  layer_num = 0
  for param_name in net.params.keys():
    layer_num += 1
    param = net.params[param_name][0]
    data = param.data
    shape = data.shape
    squeezed = np.squeeze(data)
    padsize=1
    if len(squeezed.shape) > 3:
      for n in xrange(squeezed.shape[0]):
        d = squeezed[n]
        while len(d.shape) < 3:
          d = np.expand_dims(d, axis=0)
          padsize=0
        title = '%s Param=%s Shape=%s Num=%s (%.3f, %.3f, %.3f)'\
                %(iteration, param_name, shape, n, np.min(d), np.mean(d), np.max(d))
        fname = 'param%i_%s_n%i%s.png'%(layer_num,param_name,n,cnt)
        print '[X-Ray] Saving Param:', param_name, 'Num =', n, 'Shape =',shape
        vis_square(d, title=title, fname=join(save_dir,fname),
                   padsize=padsize, padval=1)
    else:
      while len(squeezed.shape) < 3:
        squeezed = np.expand_dims(squeezed, axis=0)
        padsize=0
      title = '%s Param=%s Shape=%s (%.3f, %.3f, %.3f)'\
              %(iteration, param_name, shape, np.min(data), np.mean(data), np.max(data))
      fname = 'param%i_%s%s.png'%(layer_num,param_name,cnt)
      print '[X-Ray] Saving Param:', param_name, 'Shape =', shape
      vis_square(squeezed, title=title, fname=join(save_dir,fname),
                 padsize=padsize, padval=1)

def xray(net, save_dir):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  param_dir = join(save_dir, 'params')
  if not os.path.exists(param_dir):
    os.makedirs(param_dir)
  save_params(net, save_dir=param_dir)

  blob_dir = join(save_dir, 'blob_data')
  if not os.path.exists(blob_dir):
    os.makedirs(blob_dir)
  save_blobs(net, save_dir=blob_dir, use_diff=False)

  blob_diff = join(save_dir, 'blob_diff')
  if not os.path.exists(blob_diff):
    os.makedirs(blob_diff)
  save_blobs(net, save_dir=blob_diff, use_diff=True)
  # Visualize the maximizing patches
  # patch_dir = join(save_dir,'maximizing_patches')
  # os.makedirs(patch_dir)
  # save_maximizing_patches('conv1_layer','conv1', image_dir, patch_dir)
  # save_maximizing_patches('conv2_layer','conv2', image_dir, patch_dir)

def fmri(snapshot_prefix, save_dir, net_prototxt='drqn.prototxt', phase=caffe.TRAIN):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  movie_dir = join(save_dir, 'movie')
  if not os.path.exists(movie_dir):
    os.makedirs(movie_dir)
  import subprocess
  import glob
  for i, caffemodel in enumerate(glob.glob(snapshot_prefix + '*.caffemodel')):
    iteration = caffemodel.split('_iter_')[-1].split('.caffemodel')[0]
    net = caffe.Net(net_prototxt, caffemodel, phase)
    save_params(net, save_dir=movie_dir, iteration=iteration, cnt=i)
  for prefix in list(set(['_'.join(f.split('_')[:-1])
                          for f in glob.glob(join(movie_dir,'*.png'))])):
    cmd = 'ffmpeg -framerate 1 -i %s_%%02d.png -r 30 -pix_fmt yuv420p %s.mp4'%(prefix,prefix)
    subprocess.check_call(cmd.split(' '))

# Use scipy.optimize to find the optimal input to maximize a convolutional filter
def optimize_filter(layer_name, filter_num=0):
  from scipy.optimize import minimize
  shape = net.params[layer_name][0].data[filter_num].shape
  x0 = np.zeros(shape)
  conv_params = net.params[layer_name][0].data[filter_num]
  bias = net.params[layer_name][1].data.flatten()[filter_num]
  fun = lambda x: -(np.inner(x, conv_params.flatten()) + bias)
  res = minimize(fun, x0)
  return res.x.reshape(shape).astype(np.float32), res

# Returns the input pixel region that generated the given layer's activation
# location: (y, x) or (y_min, y_max, x_min, x_max)
def get_input_patch(layer_name, location):
  if len(location) == 2:
    location = (location[0], location[0], location[1], location[1])
  if layer_name == 'conv1_layer':
    return get_lower_layer_patch(location, stride=4, kernel_size=8)
  elif layer_name == 'conv2_layer':
    conv1_patch = get_lower_layer_patch(location, stride=2, kernel_size=4)
    return get_input_patch('conv1_layer', conv1_patch)
  else:
    raise Exception('Layer Not Supported')

# to_patch: region of (y_min, y_max, x_min, x_max)
def get_lower_layer_patch(to_patch, stride, kernel_size):
  y_min, y_max, x_min, x_max = to_patch
  input_y_min, input_x_min = (y_min * stride, x_min * stride)
  input_y_max, input_x_max = (y_max * stride + kernel_size,
                              x_max * stride + kernel_size)
  return (input_y_min, input_y_max, input_x_min, input_x_max)

# Locate the image that maximizes the activation of a given unit
def save_maximizing_patches(layer_name, blob_name, image_dir, save_dir,
                            stride=4, kernel_size=8, pad=0):
  assert layer_name in net.params
  assert blob_name in net.blobs

  assert os.path.isdir(image_dir)
  batch_size = net.blobs['frames'].data.shape[0]
  files = [f for f in os.listdir(image_dir) if isfile(join(image_dir,f)) ]
  def load_frame_data(fname):
    return np.fromfile(open(fname,'rb'), dtype=np.uint8) \
             .reshape(4,84,84).astype(np.float32)
  n_filters = net.params[layer_name][0].data.shape[0]
  max_activations = np.zeros(n_filters).astype(np.float32)
  max_activations.fill('-inf')
  patches = [None] * n_filters
  locations = [None] * n_filters
  reference_images = [None] * n_filters
  for batch in xrange(int(len(files) / batch_size)):
    image_batch = []
    for i in xrange(batch_size):
      fname = join(image_dir, files[batch * batch_size + i])
      image_batch.append(load_frame_data(fname))
    input_frames = np.asarray(image_batch)
    forward_from_frames(input_frames)
    for filter in xrange(n_filters):
      filter_act = net.blobs[blob_name].data[:,filter,:,:]
      max_act = np.max(filter_act)
      if max_act > max_activations[filter]:
        max_activations[filter] = max_act
        max_loc = np.unravel_index(filter_act.argmax(), filter_act.shape)
        assert max_activations[filter] == filter_act[max_loc]
        max_n, max_y, max_x = max_loc
        print 'Found max act %f for filter %d'%(max_act, filter)
        reference_images[filter] = np.copy(input_frames[max_n])
        # Which region of the input image generated this activation?
        input_y, input_x = (max_y * stride - pad, max_x * stride - pad)
        in_patch = get_input_patch(layer_name, (max_y, max_x))
        locations[filter] = in_patch
        patch = input_frames[max_n,:,in_patch[0]:in_patch[1], in_patch[2]:in_patch[3]]
        print 'Input Patch:', locations[filter]
        patches[filter] = patch
        # Double check the convolution
        # conv_params = net.params[layer_name][0].data[filter]
        # bias = net.params[layer_name][1].data.flatten()[filter]
        # act_check = np.inner(patch.flatten(), conv_params.flatten()) + bias
        # assert np.allclose(act_check, max_act)
  for filter in xrange(n_filters):
    fname = join(save_dir, '%s_filter%d.png'%(layer_name, filter))
    vis_filter(layer_name, filter, fname=fname)
    title = '[MaxPatch] Layer=%s FilterNum=%d Activation=%0.f (y,x)=(%d:%d, %d:%d)'\
            %(layer_name, filter, max_activations[filter],
              locations[filter][0], locations[filter][1],
              locations[filter][2], locations[filter][3])
    fname = join(save_dir, '%s_filter%d_maxact.png'%(layer_name, filter))
    vis_square(patches[filter], padval=1, title=title, fname=fname)
    fname = join(save_dir, '%s_filter%d_reference_frames.png'%(layer_name, filter))
    vis_square(reference_images[filter], padval=1, title='Reference Image', fname=fname)

def test():
  vis_filter('conv1_layer', 10, fname='f10.png')
  opt, res = optimize_filter('conv1_layer', 10)
  vis_square(opt, title='Optimized inputs', fname='opt10.png')

seed(123)
prototxt = 'gen_net.prototxt'
snapshot = 'state/singleTower_cheat_lrmult05_u2f2_pong_iter_1000000.caffemodel'
snapshot_prefix = snapshot.split('_iter_')[0]
phase = caffe.TRAIN
net = caffe.Net(prototxt, snapshot, phase)
image_dir = 'screen/'
save_dir = 'pong_xray/'
print 'net.blobs:'
for k, v in net.blobs.items():
  d = v.data
  print k, d.shape
print 'net.params:'
for k, v in net.params.items():
  w = v[0].data
  b = v[1].data
  print k, '[Weights]', w.shape, '[bias]', b.shape
input_layer_name = net.blobs.keys()[0]
frames = run_forward(image_dir, input_layer_name)
xray(net, save_dir)
# fmri(snapshot_prefix, save_dir, net_prototxt=prototxt, phase=caffe.TEST)
