#!/usr/bin/ipython -i
import os
from os.path import join, isfile
import numpy as np
import matplotlib.pyplot as plt
import sys
from random import choice
caffe_root = '/u/mhauskn/projects/muupan_caffe/'
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

def run_forward(image_dir):
  assert os.path.isdir(image_dir)
  batch_size = net.blobs['frames'].data.shape[0]
  files = [f for f in os.listdir(image_dir) if isfile(join(image_dir,f)) ]
  images = []
  def load_frame_data(fname):
    return np.fromfile(open(fname,'rb'), dtype=np.uint8)\
             .reshape(4,84,84).astype(np.float32)
  for i in xrange(batch_size):
    fname = join(image_dir, choice(files))
    images.append(load_frame_data(fname))
  input_frames = np.asarray(images)
  return forward_from_frames(input_frames)

def forward_from_frames(input_frames):
  batch_size = net.blobs['frames'].data.shape[0]
  assert input_frames.shape == (32, 4, 84, 84)
  targets = np.zeros([batch_size,18,1,1], dtype=np.float32)
  filters = np.zeros([batch_size,18,1,1], dtype=np.float32)
  net.set_input_arrays(0, input_frames, np.zeros([batch_size,1,1,1], dtype=np.float32))
  net.set_input_arrays(1, targets, np.zeros([batch_size,1,1,1], dtype=np.float32))
  net.set_input_arrays(2, filters, np.zeros([batch_size,1,1,1], dtype=np.float32))
  net.forward()
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
  vis_square(activations, title=title, fname=fname)

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
  weights = net.params[layer_name][0].data[0]
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

def xray_dqn(save_dir, image_dir):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  # Visualize the conv layer weights + biases
  for layer in ['conv1_layer', 'conv2_layer']:
    for i in xrange(net.params[layer][0].data.shape[1]):
      vis_dim(layer, i, join(save_dir, layer + '_dim' + str(i) +'.png'))
    vis_mean_filters(layer, join(save_dir, layer + '_mean.png'))
    vis_biases(layer, join(save_dir, layer+'_biases.png'))
  # Visualize fc layer weights + biases
  for layer in ['ip1_layer', 'ip2_layer']:
    vis_weights(layer, fname=join(save_dir, layer + '_weights.png'))
    vis_biases(layer, join(save_dir, layer + '_biases.png'))
  # Visualize the Inputs
  frames = run_forward(image_dir)
  title = '[Input] Blob=%s Num=%d (%.3f,%.3f,%.3f)'\
          %('frames', 0, np.min(frames), np.mean(frames), np.max(frames))
  vis_square(frames[0], padval=1, title=title,
             fname=join(save_dir,'input_activations.png'))
  # Visualize the activations
  for blob_name in ['conv1', 'conv2', 'ip1', 'q_values']:
    vis_activations(blob_name, fname=join(
      save_dir, blob_name + '_activations.png'))
  # Visualize the most active FC-1 nodes
  sorted_activations = np.argsort(net.blobs['ip1'].data[0].flatten())[::-1]
  for i in xrange(5):
    idx = sorted_activations[i]
    activation = net.blobs['ip1'].data[0].flatten()[idx]
    vis_fc_incoming_weights('ip1_layer', activation, [32,9,9],
                            fname=join(save_dir,'ip1_unit'+str(idx)+'.png'),
                            unit=idx)

# Find the optimal input to maximize a convolutional filter
def optimize_filter(layer_name, filter_num=0):
  from scipy.optimize import minimize
  shape = net.params[layer_name][0].data[filter_num].shape
  x0 = np.zeros(shape)
  conv_params = net.params[layer_name][0].data[filter_num]
  bias = net.params[layer_name][1].data.flatten()[filter_num]
  fun = lambda x: -(np.inner(x, conv_params.flatten()) + bias)
  res = minimize(fun, x0)
  return res.x.reshape(shape).astype(np.float32), res

# Locate the image that maximizes the activation of a given unit
def find_optimizing_image(layer_name, blob_name, image_dir, save_dir,
                          stride=4, kernel_size=8, pad=0):
  assert layer_name in net.params
  assert blob_name in net.blobs
  assert pad == 0
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
        locations[filter] = (input_y, input_x)
        # print 'Input Region (%d:%d,%d:%d)'\
        #   %(input_y,input_y+kernel_size,input_x,input_x+kernel_size)
        patch = input_frames[max_n,:,input_y:input_y+kernel_size,
                             input_x:input_x+kernel_size]
        # Double check the convolution
        conv_params = net.params[layer_name][0].data[filter]
        bias = net.params[layer_name][1].data.flatten()[filter]
        act_check = np.inner(patch.flatten(), conv_params.flatten()) + bias
        assert np.allclose(act_check, max_act)
        patches[filter] = patch
  for filter in xrange(n_filters):
    fname = join(save_dir, '%s_filter%d.png'%(layer_name, filter))
    vis_filter(layer_name, filter, fname=fname)
    title = '[MaxPatch] Layer=%s FilterNum=%d Activation=%0.f (y,x)=(%d:%d, %d:%d)'\
            %(layer_name, filter, max_activations[filter],
              locations[filter][0], locations[filter][0] + kernel_size,
              locations[filter][1], locations[filter][1] + kernel_size)
    fname = join(save_dir, '%s_filter%d_maxact.png'%(layer_name, filter))
    vis_square(patches[filter], padval=1, title=title, fname=fname)
    fname = join(save_dir, '%s_filter%d_reference_frames.png'%(layer_name, filter))
    vis_square(reference_images[filter], padval=1, title='Reference Image', fname=fname)

def test():
  vis_filter('conv1_layer', 10, fname='f10.png')
  opt, res = optimize_filter('conv1_layer', 10)
  vis_square(opt, title='Optimized inputs', fname='opt10.png')

if len(sys.argv) < 3:
  raise Exception('usage: load_net.py net.prototxt snapshot.caffemodel')
else:
  net = caffe.Net(sys.argv[1], sys.argv[2])
  net.set_phase_test()
  net.set_mode_cpu()
  print 'net.blobs:'
  for k, v in net.blobs.items():
    print k, v.data.shape
  print 'net.params:'
  for k, v in net.params.items():
    print (k, v[0].data.shape)
