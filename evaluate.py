import sys
import os
import numpy as np
import tensorflow as tf
import gzip
import cPickle
import random
from config import FLAGS
from models import DynamicRNN
from models import AveragePooling
from sklearn.metrics import average_precision_score

TEST = 'test.pkl.gz'
with gzip.open(TEST, 'r') as f:
  data = cPickle.load(f)

model = AveragePooling(feature_size=FLAGS.feature_size, max_video_length=FLAGS.max_video_length,
    num_classes=FLAGS.num_classes, cell_size=FLAGS.size, use_lstm=FLAGS.use_lstm,
    learning_rate=None, learning_rate_decay_factor=None, min_learning_rate=None, 
    training_steps_per_epoch=None, max_gradient_norm=None, keep_prob=1.0, is_training=False)

with tf.Session() as sess:
  start_step = 200
  for step in range(start_step, 1000000, FLAGS.steps_per_checkpoint):
    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "ckpt-%d"%step)
    model.saver.restore(sess, checkpoint_path)
    predictions, labels = [], []
    for start, end in zip(range(0, len(data), FLAGS.batch_size), range(FLAGS.batch_size, len(data), FLAGS.batch_size)):
      batch_data = data[start:end]
      batch_feature = []
      batch_label = []
      batch_length = []
      for vid, feature, label in batch_data:
        batch_length.append(feature.shape[0])
        batch_label.append(label)
        labels.append(label)
        pad_length = FLAGS.max_video_length - feature.shape[0]
        feature = np.concatenate((feature, np.zeros((pad_length, FLAGS.feature_size), dtype=np.float32)),axis=0)
        batch_feature.append(feature)
      feed_dict = {model.frame_feature_ph: batch_feature, model.video_length_ph:batch_length, model.video_label_ph:batch_label}

      prediction = sess.run([model.logit], feed_dict=feed_dict)[0].tolist()
      predictions += prediction
    mAP = average_precision_score(labels, predictions)
    print '%5d: %.3f'%(step, mAP)
    sys.stdout.flush()
