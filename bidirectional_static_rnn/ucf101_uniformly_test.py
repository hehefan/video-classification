import sys
import os
import numpy as np
import tensorflow as tf
from config import FLAGS
from model import BiStaticRNN

ClassID = {}
with open('../data/ucfTrainTestlist/classInd.txt', 'r') as f:
  for line in f:
    class_id, class_name = line.split()
    ClassID[class_name.strip()] = int(class_id) - 1

video_label = []
with open('../data/ucfTrainTestlist/testlist01.txt', 'r') as f:
  for line in f:
    class_name, video_name = line.split('/')
    video_label.append((video_name.strip(), ClassID[class_name.strip()]))

FEATURE_HOME = '../feature-extraction/ucf101-global-pool-2048-features'
dataset = []
for video_name, class_id in video_label:
  video_feature_name = video_name.split('.')[0] + '.npy'
  video_feature_path = os.path.join(FEATURE_HOME, video_feature_name)
  video_feature = np.load(video_feature_path)
  video_length = video_feature.shape[0]
  if video_length < FLAGS.video_length:
    pad = np.stack([np.zeros(FLAGS.feature_size)]*(FLAGS.video_length-video_length))
    video_feature = np.concatenate((pad, video_feature), axis=0)
  elif video_length > FLAGS.video_length:
    t = video_length/float(FLAGS.video_length)
    idx = 0
    features = []
    for _ in range(FLAGS.video_length):
      features.append(video_feature[int(round(idx))])
      idx += t
    video_feature = np.stack(features)
  dataset.append((video_feature, class_id))

model = BiStaticRNN(feature_size=FLAGS.feature_size,
                    num_layers=FLAGS.num_layers,
                    video_length=FLAGS.video_length,
                    num_classes=FLAGS.num_classes,
                    cell_size=FLAGS.cell_size,
                    use_lstm=FLAGS.use_lstm)

with tf.Session() as sess:
  start_step = FLAGS.steps_per_checkpoint
  max_acc = 0
  for step in range(start_step, 99999999999, FLAGS.steps_per_checkpoint):
    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "ckpt-%d"%step)
    if not os.path.exists(checkpoint_path+'.index'):
      break
    else:
      model.saver.restore(sess, checkpoint_path)
      cnt = 0.0
      for feature, label in dataset:
        batch_label = [label]
        batch_feature = [feature]
        feed_dict = {model.frame_feature_ph: batch_feature, model.video_label_ph:batch_label}
        prediction = sess.run(model.prediction, feed_dict=feed_dict)[0]
        cnt += int(prediction == label)
      acc = cnt/len(dataset)
      print '%5d: %.6f'%(step, acc)
      sys.stdout.flush()
      if max_acc < acc:
        max_acc = acc
  print 'Max accuracy: %.6f'%max_acc
