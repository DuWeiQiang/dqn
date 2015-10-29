from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# First make the images then make a video:
# avconv -r 30 -i movie/%05d.png -f mp4 -c:v libx264 drqnFlickerPong.mp4

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def join_images(img_num, border=10, folder='screen/'):
  prefix = folder + str(img_num)
  img = Image.open(prefix + '.png').resize((210,210))
  width, height = img.size
  # print width, height
  joint_img = Image.new('RGB', (width + 210 + border, height), 'white')
  joint_img.paste(img, (0,0))
  data = np.fromfile(open(prefix + '.bin','rb'), dtype=np.uint8).reshape(84,84).astype(np.float32)
  data = data[1:-8,:]
  if not np.any(data):
    data += 79

  #   data[0,:] = 223
  #   data[-8,:] = 162
  #   data[-7:,:] = 232
  binary_img = Image.fromarray(data).resize((210,210))
  joint_img.paste(binary_img, (width + border,0))
  return joint_img

import os
for i in xrange(len(os.listdir('screen/'))/2):
  img = join_images(i)
  fname = 'movie/%05i.png'%i
  img.save(fname)
