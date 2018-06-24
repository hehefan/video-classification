import tensorflow as tf

class NonParametricModel(object):
  def __init__(self, support_set):

    self.support_set = tf.constant(support_set)
    self.video_feature_ph = tf.placeholder(tf.float32, [1, support_set.shape[1]])

    logits = tf.matmul(a=self.support_set, b=self.video_feature_ph, transpose_b=True)
    self.prediction = tf.argmax(logits, 0)
