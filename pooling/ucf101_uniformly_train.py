import sys
import os
import numpy as np
import tensorflow as tf
from config import FLAGS
from model import FullyConnection

video_label = []
with open('../data/ucfTrainTestlist/trainlist01.txt', 'r') as f:
  for line in f:
    video_info, class_id = line.split()
    _, video_name = video_info.split('/')
    video_label.append((video_name.strip(), int(class_id) - 1)) 

FEATURE_HOME = '../feature-extraction/ucf101-global-pool-2048-features'
dataset = []
for video_name, class_id in video_label:
  video_feature_name = video_name.split('.')[0] + '.npy'
  video_feature_path = os.path.join(FEATURE_HOME, video_feature_name)
  video_feature = np.load(video_feature_path)
  video_length = video_feature.shape[0]
  if video_length > FLAGS.video_length:
    t = video_length/float(FLAGS.video_length)
    idx = 0
    features = []
    for _ in range(FLAGS.video_length):
      features.append(video_feature[int(round(idx))])
      idx += t
    video_feature = np.stack(features)
  dataset.append((video_feature, class_id))

training_steps_per_epoch = len(dataset) // FLAGS.batch_size
model = FullyConnection(feature_size=FLAGS.feature_size,
                        num_classes=FLAGS.num_classes,
                        learning_rate=FLAGS.learning_rate,
                        learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                        min_learning_rate=FLAGS.min_learning_rate,
                        training_steps_per_epoch=training_steps_per_epoch,
                        keep_prob=FLAGS.keep_prob,
                        is_training=True)

with tf.Session() as sess:
  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(sess, ckpt.model_checkpoint_path)
    step = int(ckpt.model_checkpoint_path.split('-')[1])
  else:
    sess.run(tf.global_variables_initializer())
    step = 0
  for epoch in range(1, FLAGS.num_epochs+1):
    np.random.shuffle(dataset)
    for start, end in zip(range(0, len(dataset), FLAGS.batch_size), range(FLAGS.batch_size, len(dataset), FLAGS.batch_size)):
      batch_feature = []
      batch_label = []
      for i in range(start, end):
        feature, label = dataset[i]
        # feature = np.mean(feature, axis=0)  # average pooling
        feature = np.max(feature, axis=0)  # max pooling
        batch_label.append(label)
        batch_feature.append(feature)
      feed_dict = {model.video_feature_ph: batch_feature, model.video_label_ph:batch_label}
      loss, learning_rate, _ = sess.run([model.loss, model.learning_rate, model.train_op], feed_dict=feed_dict)
      step += 1
      if step % FLAGS.steps_per_checkpoint == 0:
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
      print "%5d: %.6f\t%.6f"%(step, learning_rate, loss)
      sys.stdout.flush()
