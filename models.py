import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn

class AveragePooling(object):
  def __init__(self, feature_size, max_video_length, num_classes, cell_size, use_lstm, learning_rate,
      learning_rate_decay_factor, min_learning_rate, training_steps_per_epoch, max_gradient_norm,
      keep_prob=0.5, is_training=False):

    self.frame_feature_ph = tf.placeholder(tf.float32, [None, max_video_length, feature_size])
    self.video_length_ph = tf.placeholder(tf.int32, [None])
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

    video_length_ph = tf.cast(x=self.video_length_ph, dtype=tf.float32)
    video_length_ph = tf.stack(values=[video_length_ph]*feature_size, axis=1)
    state = tf.reduce_sum(input_tensor=self.frame_feature_ph, axis=1) / video_length_ph
    state = tf.nn.relu(state)

    if is_training:
      state = tf.nn.dropout(state, keep_prob=keep_prob)

    if num_classes == 2:
      with tf.variable_scope('Classification'):
        logit = tf.contrib.layers.fully_connected(inputs=state, num_outputs=1, activation_fn=None) # [batch_size, 1]
      self.logit = tf.squeeze(logit)                                                               # [batch_size]
      if is_training:
        video_label = tf.cast(x=self.video_label_ph, dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=video_label, logits=self.logit))
      else:
        self.prediction = tf.cast(tf.greater(x=logit, y=0.5), tf.int32)
    else:
      with tf.variable_scope('Classification'):
        self.logits = tf.contrib.layers.fully_connected(inputs=state, num_outputs=num_classes, activation_fn=None) # [batch_size, num_classes]
      if is_training:
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.video_label_ph, logits=self.logits))
      else:
        self.prediction = tf.argmax(logits, 1)
        
    if is_training:
      params = tf.trainable_variables()
      gradients = tf.gradients(self.loss, params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
        zip(clipped_gradients, params), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)

class DynamicRNN(object):
  def __init__(self, feature_size, max_video_length, num_classes, cell_size, use_lstm, learning_rate,
      learning_rate_decay_factor, min_learning_rate, training_steps_per_epoch, max_gradient_norm,
      keep_prob=0.5, is_training=False):

    self.frame_feature_ph = tf.placeholder(tf.float32, [None, max_video_length, feature_size])
    self.video_length_ph = tf.placeholder(tf.int32, [None])
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

    # Make RNN cells
    cell = GRUCell(cell_size)
    if use_lstm:
      cell = BasicLSTMCell(cell_size, state_is_tuple=False)

    # RNN
    with tf.variable_scope('DynamicRNN'):
      outputs, state = dynamic_rnn(cell=cell, inputs=self.frame_feature_ph, sequence_length=self.video_length_ph, dtype=tf.float32)

    state = tf.nn.relu(state)

    if is_training:
      state = tf.nn.dropout(state, keep_prob=keep_prob)

    if num_classes == 2:
      with tf.variable_scope('Classification'):
        logit = tf.contrib.layers.fully_connected(inputs=state, num_outputs=1, activation_fn=None) # [batch_size, 1]
      self.logit = tf.squeeze(logit)                                                               # [batch_size]
      if is_training:
        video_label = tf.cast(x=self.video_label_ph, dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=video_label, logits=self.logit))
      else:
        self.prediction = tf.cast(tf.greater(x=logit, y=0.5), tf.int32)
    else:
      with tf.variable_scope('Classification'):
        self.logits = tf.contrib.layers.fully_connected(inputs=state, num_outputs=num_classes, activation_fn=None) # [batch_size, num_classes]
      if is_training:
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.video_label_ph, logits=self.logits))
      else:
        self.prediction = tf.argmax(logits, 1)
        
    if is_training:
      params = tf.trainable_variables()
      gradients = tf.gradients(self.loss, params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
        zip(clipped_gradients, params), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)
