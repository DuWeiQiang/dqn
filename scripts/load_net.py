#!/usr/bin/ipython -i
import os
from os.path import join, isfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import sys
from random import seed, choice
caffe_root = '/u/mhauskn/projects/recurrent_caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

fnum = 0

def load_frame_data(fname):
  return np.fromfile(open(fname,'rb'), dtype=np.uint8).reshape(84,84).astype(np.float32)

# Test_mode will load images sequentially only into blob num zero. All
# other images will be filled with ones.
def run_forward(image_dir=None, input_layer_name=None, test_mode=False, first_frame=False):
  global fnum
  if not input_layer_name:
    input_layer_name = net.blobs.keys()[0]
  assert input_layer_name in net.blobs.keys()
  n,c,w,h = net.blobs[input_layer_name].data.shape
  assert w == 84 and h == 84
  if image_dir is not None:
    assert os.path.isdir(image_dir)
    files = [f for f in os.listdir(image_dir) if isfile(join(image_dir,f)) ]
  images = []
  for i in xrange(n):
    frames = []
    for j in xrange(c):
      if image_dir is None or (i > 0 and test_mode):
        frames.append(np.ones([84,84], dtype=np.float32))
      else:
        if fnum >= len(files):
          fnum = 0
        frames.append(load_frame_data(join(image_dir, files[fnum])))
        fnum += 1
    joined_frames = np.asarray(frames)
    images.append(joined_frames)
  input_frames = np.asarray(images)
  return forward_from_frames(input_frames, input_layer_name, first_frame)

def forward_from_frames(input_frames, input_layer_name, first_frame=False):
  n,c,w,h = net.blobs[input_layer_name].data.shape
  layer_types = [x.type for x in net.layers]
  mem_layer_indices = [i for i, x in enumerate(layer_types) if x == 'MemoryData']
  assert len(mem_layer_indices) >= 2, 'Expected at least 2 memory layers!'
  kUnroll = net.blobs['cont'].data.shape[0]
  if len(mem_layer_indices) > 0:
    input_idx = mem_layer_indices[0]
    input_labels = np.zeros([n,1,1,1], dtype=np.float32)
    net.set_input_arrays(input_idx, input_frames, input_labels)
  if len(mem_layer_indices) > 1:
    cont_idx = mem_layer_indices[1]
    cont_data = np.ones_like(net.blobs['cont'].data, dtype=np.float32)
    if first_frame:
      print 'Zeroing cont for first frame'
      cont_data *= 0 # Zero cont on the first frame
    cont_labels = np.zeros([kUnroll,1,1,1], dtype=np.float32)
    net.set_input_arrays(cont_idx, cont_data, cont_labels)
  if len(mem_layer_indices) > 2:
    target_idx = mem_layer_indices[2]
    target_data = np.zeros_like(net.blobs['target'].data, dtype=np.float32)
    target_labels = np.zeros([kUnroll,1,1,1], dtype=np.float32)
    net.set_input_arrays(target_idx, target_data, target_labels)
  if len(mem_layer_indices) > 3:
    filter_idx = mem_layer_indices[3]
    filter_data = np.ones_like(net.blobs['filter'].data, dtype=np.float32)
    filter_labels = np.zeros([kUnroll,1,1,1], dtype=np.float32)
    net.set_input_arrays(filter_idx, filter_data, filter_labels)
  net.forward()
  net.backward()
  return input_frames

# Input data is an array of shape (n, height, width)
def superimpose(input_data, normalize=True):
  data = np.copy(input_data)
  n,h,w = data.shape
  coefs = np.linspace(0, 1, num=n+1)[1:]
  img = np.zeros([h,w])
  for i in xrange(n):
    # Ignore missing screens
    if len(np.unique(data[i])) > 1:
      img += coefs[i] * coefs[i] * data[i]
  if normalize:
    img -= img.min()
    img /= img.max()
  return img

# Visualize a sequence of frames by interposing them on each other
# using recency weighting. Input data is an array of shape (n, height, width)
def superimpose_plot(input_data, title='', fname='', rect=None):
  # Crop the frame a little to remove irrelevant areas
  img = superimpose(input_data[:,5:73,:])
  plt.clf()
  currentAxis = plt.gca()
  if rect is not None:
    x,y,width,height = rect
    currentAxis.add_patch(Rectangle((x, y-5), width, height, fill=None, edgecolor='r'))

  plt.imshow(img)
  plt.title(title)
  if not fname:
    plt.show()
  else:
    plt.savefig(fname)

# Images is an array of shape (n, height, width)
def vis_grid(images, n_cols=None, title=None, fname=None, rects=None, padsize=1,
             padval=0, acts=None):
  n, h, w = images.shape
  if not n_cols:
    n_cols = n
  n_rows = n / n_cols
  img = np.ones([n_rows * (h + padsize), n_cols * (w + padsize)]) * padval
  vpad = 0 if n_rows == 1 else padsize
  for i in xrange(n):
    row = i / n
    col = i % n
    img[row * (h + vpad) : (row + 1) * h + row * vpad,
        col * (w + padsize) : (col + 1) * w + col * padsize] = images[i]
  plt.clf()
  fig = plt.figure(frameon=False)
  # dpi = 600
  fig.set_size_inches(img.shape[1]/50.0,img.shape[0]/50.0)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)

  currentAxis = plt.gca()
  if rects is not None:
    for i, rect in enumerate(rects):
      x,y,width,height = rect
      row = i / n
      col = i % n
      currentAxis.add_patch(Rectangle((col * (w + padsize) + x, row * (h + vpad) + y), width, height, fill=None, edgecolor='r'))

  ax.imshow(img, aspect='auto')

  # Plot activations over the images
  if acts is not None:
    assert len(acts) == n
    ax.autoscale(False)
    ax.annotate(str(round(np.max(acts),3)), xy=(3, 1),  xycoords='data',
                xytext=(0.07, 0.9), textcoords='axes fraction')
    ax.annotate(str(round(np.min(acts),3)), xy=(3, 1),  xycoords='data',
                xytext=(0.07, 0.02), textcoords='axes fraction')
    acts *= -1 # Flip data due to axes being switched
    # Normalize and scale into the picture
    acts -= acts.min()
    acts /= acts.max()
    acts *= img.shape[0] - 2
    acts += 1
    x = [(w+1)+w*z+(padsize)*(z-1) for z in xrange(n)]
    ax.plot(x, acts, marker='o', ls='-')

  fig.savefig(fname)#, dpi=dpi)

  # plt.imshow(img)
  # if title is not None:
  #   plt.title(title)
  # plt.axis('off')
  # if not fname:
  #   plt.show()
  # else:
  #   plt.savefig(fname, bbox_inches='tight', pad_inches=0)

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(input_data, padsize=1, padval=0, title='', fname='', rect=None):
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
  # plt.figure()
  currentAxis = plt.gca()
  plt.imshow(data)
  if rect is not None:
    x,y,width,height = rect
    for i in xrange(input_data.shape[0]):
      new_x = (i % n) * (input_data.shape[2] + padsize) + x
      new_y = (i / n) * (input_data.shape[1] + padsize) + y
      currentAxis.add_patch(Rectangle((new_x, new_y), width, height, fill=None,
                                      edgecolor='r'))
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

def save_blobs(net, save_dir, to_save, use_diff=False):
  blob_num = 0
  for blob_name in to_save.keys():
    blob_num += 1
    blob = net.blobs[blob_name]
    data = blob.diff if use_diff else blob.data
    shape = data.shape
    squeezed = np.squeeze(data)
    padsize=1
    if len(squeezed.shape) > 2:
      for n in to_save[blob_name]:
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

def xray(net, save_dir, xray_params=True, xray_data=[], xray_diff=[]):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if xray_params:
    param_dir = join(save_dir, 'params')
    if not os.path.exists(param_dir):
      os.makedirs(param_dir)
    save_params(net, save_dir=param_dir)

  if xray_data:
    blob_dir = join(save_dir, 'blob_data')
    if not os.path.exists(blob_dir):
      os.makedirs(blob_dir)
    save_blobs(net, blob_dir, xray_data, use_diff=False)

  if xray_diff:
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
# Returns: (y_min, y_max, x_min, x_max)
def get_input_patch(layer_name, location):
  if len(location) == 2:
    location = (location[0], location[0], location[1], location[1])
  if layer_name == 'conv1':
    return get_lower_layer_patch(location, stride=4, kernel_size=8)
  elif layer_name == 'conv2':
    conv1_patch = get_lower_layer_patch(location, stride=2, kernel_size=4)
    return get_input_patch('conv1', conv1_patch)
  elif layer_name == 'conv3':
    conv2_patch = get_lower_layer_patch(location, stride=1, kernel_size=3)
    return get_input_patch('conv2', conv2_patch)
  else:
    raise Exception('Layer Not Supported')

# Get the patch of the lower conv layer that generated to_patch
# to_patch: region of (y_min, y_max, x_min, x_max)
def get_lower_layer_patch(to_patch, stride, kernel_size):
  y_min, y_max, x_min, x_max = to_patch
  input_y_min, input_x_min = (y_min * stride, x_min * stride)
  input_y_max, input_x_max = (y_max * stride + kernel_size,
                              x_max * stride + kernel_size)
  return (input_y_min, input_y_max, input_x_min, input_x_max)

# Save lstm activations to disk as a numpy array of dimension: [timesteps, units]
def save_lstm_activations(image_dir, save_dir, layer_name='lstm1', blob_name='lstm1',
                          input_layer_name='all_frames', test_mode=True):
  global fnum
  assert layer_name in net.params
  assert blob_name in net.blobs
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  assert os.path.isdir(image_dir)
  assert blob_name.startswith('lstm') and layer_name.startswith('lstm')
  files = [f for f in os.listdir(image_dir) if isfile(join(image_dir,f)) ]
  n,c,w,h = net.blobs[input_layer_name].data.shape
  # acts = np.zeros([len(files), net.blobs[blob_name].data.shape[2]])
  # blank = np.zeros(len(files))
  acts = np.load(join(save_dir, 'acts.npy'))
  blank = np.load(join(save_dir, 'blank.npy'))
  fnum = int(np.load(join(save_dir, 'fnum.npy')))
  for ts in xrange(fnum, len(files)):
    input_frames = run_forward(image_dir, test_mode=test_mode, first_frame=ts==0)
    fnum -= 9
    acts[ts,:] = net.blobs[blob_name].data[0,0,:]
    if len(np.unique(input_frames[0,-1])) == 1:
      blank[ts] = 1
    sys.stderr.write('TS = %i, blank = %i, fnum = %i\n'%(ts, blank[ts], fnum))
    if ts % 100 == 0:
      sys.stderr.write('saving activations\n')
      np.save(join(save_dir, 'acts.npy'), acts)
      np.save(join(save_dir, 'blank.npy'), blank)
      np.save(join(save_dir, 'fnum.npy'), fnum)

def save_seq_lstm_activation(start_fnum, unit, image_dir, save_dir,
                             unroll=10, layer_name='lstm1', blob_name='lstm1',
                             input_layer_name='all_frames', test_mode=True):
  global fnum
  fnum = start_fnum
  assert layer_name in net.params
  assert blob_name in net.blobs
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  assert os.path.isdir(image_dir)
  assert blob_name.startswith('lstm') and layer_name.startswith('lstm')
  n,c,w,h = net.blobs[input_layer_name].data.shape
  frames = []
  acts = np.zeros(unroll)
  for ts in xrange(unroll):
    input_frames = run_forward(image_dir, test_mode=test_mode, first_frame=ts==0)
    frames.append(input_frames[0,0]) # TODO: Only in test mode
    act = net.blobs[blob_name].data[:,0,unit]
    acts[ts] = act
    print 'TS =', ts, 'act = ', act
  fname = join(save_dir, 'ts%d_unit%d_act%f.png'%(start_fnum, unit, np.max(acts)))
  frames = np.asarray(frames)
  print acts
  vis_grid(frames[:,1:75,:], fname=fname, padval=1, acts=acts)

# Filter nums is a list of filter numbers
def save_topn_maximizing_lstm_patches(layer_name, blob_name, image_dir,
                                      save_dir, filter_nums=None, topn=5,
                                      input_layer_name='frames', test_mode=True):
  def get_key(act, batch, num, filter):
    return '%f_%d_%d_%d'%(act,batch,num,filter)
  from collections import deque
  import heapq
  assert layer_name in net.params
  assert blob_name in net.blobs
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  assert os.path.isdir(image_dir)
  assert blob_name.startswith('lstm') and layer_name.startswith('lstm')
  n,c,w,h = net.blobs[input_layer_name].data.shape
  files = [f for f in os.listdir(image_dir) if isfile(join(image_dir,f)) ]
  if filter_nums is None:
    filter_nums = range(net.blobs[blob_name].data.shape[2])
  n_filters = len(filter_nums)
  h,d = {},{} # Heap, data-storage
  for i in filter_nums:
    h[i] = [] # Heapqueues for each filter
    d[i] = {} # Datastorage for each filter
  frame_q = deque()
  for batch in xrange(1000): #int(len(files) / (n * c))):
    print 'Batch', batch
    input_frames = run_forward(image_dir, test_mode=test_mode)
    frame_q.append(input_frames[0,0]) # TODO: Only in test mode
    while len(frame_q) > 10:
      frame_q.popleft()
    for num in xrange(1) if test_mode else xrange(n):
      for unit in filter_nums:
        acts = net.blobs[blob_name].data[:,num,unit]
        max_act = np.max(acts)
        heap = h[unit]
        dict = d[unit]
        if len(heap) < topn or max_act > np.min([x[0] for x in heap]):
          for act, key in heap: # Remove keys that are too close in time
            ts = dict[key][2]
            if abs(ts - batch) < 10:
              if max_act > act:
                heap.remove((act,key))
                del dict[key]
          print 'Found max_act %f for unit %d'%(max_act, unit)
          # frames = np.copy(input_frames[num])
          frames = np.copy(np.asarray(frame_q))
          key = get_key(max_act, batch, num, unit)
          heapq.heappush(heap, (max_act, key))
          dict[key] = (frames, acts, batch)
        while len(heap) > topn:
          act, key = heapq.heappop(heap)
          del dict[key]
  for unit in filter_nums:
    num = 0
    heap = h[unit]
    dict = d[unit]
    f = [] # list of frames
    g = [] # list of activations
    t = [] # list of timesteps
    while heap:
      act, key = heapq.heappop(heap)
      frames, acts, ts = dict[key]
      f.append(superimpose(frames[:,1:75,:]))
      g.append(act)
      t.append(ts)
      # fname = join(save_dir, 'ts_%s_unit%d_%d_act%f.png'%(layer_name, unit, num, act))
      # vis_grid(frames[:,1:75,:], fname=fname, padval=1)
      num += 1
    fname = join(save_dir, 'si_%s_unit%d'%(layer_name, unit))
    for act,ts in zip(g,t):
      fname += '_' + str(ts) + ':' + str(round(act, 3))
    fname += '.png'
    vis_grid(np.asarray(f), fname=fname, padval=1)

# Filter nums is a list of filter numbers
def save_topn_maximizing_patches(layer_name, blob_name, image_dir,
                                 save_dir, filter_nums=None, topn=5,
                                 input_layer_name='frames'):
  def get_key(act, batch, num, filter):
    return '%f_%d_%d_%d'%(act,batch,num,filter)
  import heapq
  assert layer_name in net.params
  assert blob_name in net.blobs
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  assert os.path.isdir(image_dir)
  n,c,w,h = net.blobs[input_layer_name].data.shape
  files = [f for f in os.listdir(image_dir) if isfile(join(image_dir,f)) ]
  if filter_nums is None:
    filter_nums = range(net.params[layer_name][0].data.shape[0])
  n_filters = len(filter_nums)
  h,d = {},{} # Heap, data-storage
  for i in filter_nums:
    h[i] = [] # Heapqueues for each filter
    d[i] = {} # Datastorage for each filter
  for batch in xrange(int(len(files) / (n * c))):
    print 'Batch', batch
    input_frames = run_forward(image_dir)
    for num in xrange(n):
      for filter in filter_nums:
        filter_act = net.blobs[blob_name].data[num,filter,:,:]
        max_act = np.max(filter_act)
        heap = h[filter]
        dict = d[filter]
        if len(heap) < topn or max_act > np.min([x[0] for x in heap]):
          max_loc = np.unravel_index(filter_act.argmax(), filter_act.shape)
          max_y, max_x = max_loc
          print 'Found max act %f for filter %d at location_y,x=(%d,%d)'\
            %(max_act, filter, max_y, max_x)
          frames = np.copy(input_frames[num])
          in_patch = get_input_patch(layer_name, (max_y, max_x))
          patch = np.copy(input_frames[num,:,in_patch[0]:in_patch[1],
                                       in_patch[2]:in_patch[3]])
          key = get_key(max_act, batch, num, filter)
          heapq.heappush(heap, (max_act, key))
          dict[key] = (frames, in_patch, patch)
        while len(heap) > topn:
          act, key = heapq.heappop(heap)
          del dict[key]
  for filter in filter_nums:
    num = 0
    heap = h[filter]
    dict = d[filter]
    f = [] # list of frames
    rects = [] # list of rectangle bounding boxes
    while heap:
      act, key = heapq.heappop(heap)
      frames, loc, patches = dict[key]
      y_min, y_max, x_min, x_max = loc
      y_max = min(y_max, 75)
      x_max = min(x_max, 84)
      width = x_max - x_min
      height = y_max - y_min
      rect = (x_min, y_min - 1, width, height) # -5 due to crop
      rects.append(rect)
      f.append(superimpose(frames[:,1:75,:]))
      num += 1
    fname = join(save_dir, '%s_filter%d.png'%(layer_name, filter))
    vis_grid(np.asarray(f), fname=fname, padval=1, rects=rects)

# Locate the image that maximizes the activation of a given unit
def save_maximizing_patches(layer_name, blob_name, image_dir, save_dir,
                            stride=4, kernel_size=8, pad=0, input_layer_name='frames'):
  assert layer_name in net.params
  assert blob_name in net.blobs
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  assert os.path.isdir(image_dir)
  n,c,w,h = net.blobs[input_layer_name].data.shape
  files = [f for f in os.listdir(image_dir) if isfile(join(image_dir,f)) ]
  n_filters = net.params[layer_name][0].data.shape[0]
  max_activations = np.zeros(n_filters).astype(np.float32)
  max_activations.fill('-inf')
  patches = [None] * n_filters
  locations = [None] * n_filters
  reference_images = [None] * n_filters
  for batch in xrange(int(len(files) / (n * c))):
    print 'Batch', batch
    input_frames = run_forward(image_dir)
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
        input_y, input_x = (max_y * stride, max_x * stride)
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
    x,y,width,height = locations[filter][2], locations[filter][0], \
                       locations[filter][3] - locations[filter][2], \
                       locations[filter][1] - locations[filter][0]
    rect = (x,y,width,height)
    vis_square(reference_images[filter], padval=1, title=title, fname=fname, rect=rect)
    fname = join(save_dir, '%s_filter%d_superimposed_frames.png'%(layer_name, filter))
    superimpose(reference_images[filter], title=title, fname=fname, rect=rect)

def PrintNet(net):
  print 'net.blobs:'
  for k, v in net.blobs.items():
    d = v.data
    print k, d.shape
  print 'net.params:'
  for k, v in net.params.items():
    w = v[0].data
    b = v[1].data
    print k, '[Weights]', w.shape, '[bias]', b.shape
