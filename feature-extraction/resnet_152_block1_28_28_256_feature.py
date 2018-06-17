import os
import sys
import numpy as np
import h5py
from PIL import Image

import tensorflow as tf

from slim.nets import resnet_v1
from slim.preprocessing import vgg_preprocessing
slim = tf.contrib.slim

FRAME_HOME = '../data/ucf101-frames'
FEATURE_HOME = 'ucf101-block1-28-28-256-features'

img = tf.placeholder(dtype=tf.float32)
pre_img = vgg_preprocessing.preprocess_image(img, 224, 224, is_training=False)
pre_img = tf.expand_dims(pre_img, 0)

with slim.arg_scope(resnet_v1.resnet_arg_scope()):
	_, end_points = resnet_v1.resnet_v1_152(inputs=pre_img, is_training=False)
feature = tf.squeeze(end_points['resnet_v1_152/block1'])

if not os.path.exists('resnet_v1_152.ckpt'):
  os.system('wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz')
  os.system('tar -xvzf resnet_v1_152_2016_08_28.tar.gz')
  os.system('rm resnet_v1_152_2016_08_28.tar.gz')

if not os.path.isdir(FEATURE_HOME):
  os.mkdir(FEATURE_HOME)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  tf.train.Saver().restore(sess, './resnet_v1_152.ckpt')
  for video_name in os.listdir(FRAME_HOME):
    video_path = os.path.join(FRAME_HOME, video_name)
    feats = []
    for i in range(1, len(os.listdir(video_path))+1):
      frame_path = os.path.join(video_path, '%04d.jpg'%i)
      frame = np.array(Image.open(frame_path))
      feat = sess.run(feature, {img: frame})
      feats.append(feat)
    np.save(os.path.join(FEATURE_HOME, video_name.split('.')[0]), feats)
