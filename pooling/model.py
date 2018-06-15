import tensorflow as tf

class FullyConnection(object):
  def __init__(self,
               feature_size,
               num_classes,
               learning_rate=None,
               learning_rate_decay_factor=None,
               min_learning_rate=None,
               training_steps_per_epoch=None,
               max_gradient_norm=None,
               keep_prob=1.0,
               is_training=False):

    self.video_feature_ph = tf.placeholder(tf.float32, [None, feature_size])
    self.video_label_ph = tf.placeholder(tf.int32, [None])

    if is_training:
      self.global_step = tf.Variable(0, trainable=False)
      self.learning_rate = tf.maximum(
          tf.train.exponential_decay(
            learning_rate,
            self.global_step,
            training_steps_per_epoch,
            learning_rate_decay_factor,
            staircase=True),
          min_learning_rate)


    video_feature = tf.nn.dropout(self.video_feature_ph, keep_prob=keep_prob)

    with tf.variable_scope('Classifier'):
      logits = tf.contrib.layers.fully_connected(inputs=video_feature,
                                                 num_outputs=num_classes,
                                                 activation_fn=None) # [batch_size, num_classes]
    if is_training:
      self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.video_label_ph, logits=logits))
    else:
      self.prediction = tf.argmax(logits, 1)

    if is_training:
      params = tf.trainable_variables()
      gradients = tf.gradients(self.loss, params)
      self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
        zip(gradients, params), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
