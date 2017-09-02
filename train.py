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

TRAIN = 'train.pkl.gz'
with gzip.open(TRAIN, 'r') as f:
  data = cPickle.load(f)
training_steps_per_epoch = len(data) // FLAGS.batch_size

if not os.path.exists(FLAGS.checkpoint_dir):
  os.makedirs(FLAGS.checkpoint_dir)

model = AveragePooling(feature_size=FLAGS.feature_size, max_video_length=FLAGS.max_video_length, 
    num_classes=FLAGS.num_classes, cell_size=FLAGS.size, use_lstm=FLAGS.use_lstm, 
    learning_rate=FLAGS.learning_rate, learning_rate_decay_factor=FLAGS.learning_rate_decay_factor, 
    min_learning_rate=FLAGS.min_learning_rate, training_steps_per_epoch=training_steps_per_epoch, 
    max_gradient_norm=FLAGS.max_gradient_norm, keep_prob=FLAGS.keep_prob, is_training=True)

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
    random.shuffle(data)
    for batch_id, (start, end) in enumerate(zip(range(0, len(data), FLAGS.batch_size), range(FLAGS.batch_size, len(data), FLAGS.batch_size))):
      batch_data = data[start:end]
      batch_feature = []
      batch_label = []
      batch_length = []
      for vid, feature, label in batch_data:
        batch_length.append(feature.shape[0])
        batch_label.append(label)
        pad_length = FLAGS.max_video_length - feature.shape[0]
        feature = np.concatenate((feature, np.zeros((pad_length, FLAGS.feature_size), dtype=np.float32)),axis=0)
        batch_feature.append(feature)
      feed_dict = {model.frame_feature_ph: batch_feature, model.video_length_ph:batch_length, model.video_label_ph:batch_label}
      loss,  _ = sess.run([model.loss, model.train_op], feed_dict=feed_dict)
      step += 1
      if step % FLAGS.steps_per_checkpoint == 0:
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
      print "%5d: %3d-%3d, %.3f"%(step, epoch, batch_id+1, loss)
      sys.stdout.flush()
