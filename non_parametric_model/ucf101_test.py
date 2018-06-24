import sys
import os
import numpy as np
from sklearn.preprocessing import normalize
import tensorflow as tf
from model import NonParametricModel

FEATURE_HOME = '../feature-extraction/ucf101-global-pool-2048-features'

# training set is support set
training_video_label = []
with open('../data/ucfTrainTestlist/trainlist01.txt', 'r') as f:
  for line in f:
    video_info, class_id = line.split()
    _, video_name = video_info.split('/')
    training_video_label.append((video_name.strip(), int(class_id) - 1)) 

training_videos, training_IDs = [], []
for video_name, class_id in training_video_label:
  video_feature_name = video_name.split('.')[0] + '.npy'
  video_feature_path = os.path.join(FEATURE_HOME, video_feature_name)
  video_feature = np.max(np.load(video_feature_path), axis=0)		# max pooling
  training_videos.append(normalize(np.expand_dims(video_feature, axis=0))[0])
  #training_videos.append(video_feature)
  training_IDs.append(class_id)
training_videos = np.stack(training_videos, axis=0)             # (9537, 2048)

# test set
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

dataset = []
for video_name, class_id in video_label:
  video_feature_name = video_name.split('.')[0] + '.npy'
  video_feature_path = os.path.join(FEATURE_HOME, video_feature_name)
  video_feature = np.max(np.load(video_feature_path), axis=0)    # max pooling
  dataset.append((normalize(np.expand_dims(video_feature, axis=0))[0], class_id))
  # dataset.append((video_feature, class_id))

model = NonParametricModel(training_videos)

with tf.Session() as sess:
  cnt = 0.0
  for feature, label in dataset:
    batch_feature = [feature]
    feed_dict = {model.video_feature_ph: batch_feature}
    prediction = sess.run(model.prediction, feed_dict=feed_dict)[0]
    cnt += int(training_IDs[prediction] == label)
  acc = cnt/len(dataset)
  print '%.6f'%(acc)
  sys.stdout.flush()
