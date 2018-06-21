import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from convlstm_cell import ConvLSTMCell

class DynamicRNN(object):
  def __init__(self,
               feature_side,
               feature_size,
               num_layers,
               max_video_length,
               num_classes,
               convlstm_output_channels,
               convlstm_kernel_shape,
               conv_output_channels,
               conv_kernel_shape,
               learning_rate=None,
               learning_rate_decay_factor=None,
               min_learning_rate=None,
               training_steps_per_epoch=None,
               max_gradient_norm=None,
               keep_prob=1.0,
               input_keep_prob=1.0,
               output_keep_prob=1.0,
               state_keep_prob=1.0,
               is_training=False):

    self.frame_feature_ph = tf.placeholder(tf.float32, [None, max_video_length, feature_side, feature_side, feature_size])
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
    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell(
                  cells=[tf.contrib.rnn.DropoutWrapper(
                    cell=ConvLSTMCell(input_shape=[feature_side, feature_side, feature_size], 
                                      output_channels=convlstm_output_channels,
                                      kernel_shape=convlstm_kernel_shape),
                    input_keep_prob=input_keep_prob,
                    output_keep_prob=output_keep_prob,
                    state_keep_prob=state_keep_prob) for _ in range(num_layers)],
                  state_is_tuple=True)
    else:
      cell = ConvLSTMCell(input_shape=[feature_side, feature_side, feature_size],
                               output_channels=convlstm_output_channels,
                               kernel_shape=convlstm_kernel_shape)
      cell = tf.contrib.rnn.DropoutWrapper(
                  cell=cell,
                  input_keep_prob=input_keep_prob,
                  output_keep_prob=output_keep_prob,
                  state_keep_prob=state_keep_prob)

    # RNN
    with tf.variable_scope('DynamicRNN'):
      outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                         inputs=self.frame_feature_ph,
                                         sequence_length=self.video_length_ph,
                                         dtype=tf.float32)

    if num_layers > 1:
      state = [tf.concat(values=s, axis=3) for s in state]
    state = tf.concat(values=state, axis=3)

    with tf.variable_scope('Classifier'):
      layer = 0
      while(int(state.get_shape()[1]) != 7):
        layer += 1
        state = tf.contrib.layers.conv2d(inputs=state,
                                         num_outputs=conv_output_channels,
                                         kernel_size=conv_kernel_shape,
                                         scope='ClsConv%d'%layer)
        state = tf.layers.max_pooling2d(inputs=state,
                                        pool_size=[2,2],
                                        strides=2,
                                        padding='same',
                                        name='ClsPool%d'%layer)

      state = tf.reduce_mean(state, [1, 2], name='GlobalPool', keep_dims=False)

      state = tf.nn.dropout(state, keep_prob=keep_prob)
      logits = tf.contrib.layers.fully_connected(inputs=state,
                                                 num_outputs=num_classes,
                                                 activation_fn=None) # [batch_size, num_classes]

    if is_training:
      self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.video_label_ph, logits=logits))
    else:
      self.prediction = tf.argmax(logits, 1)

    if is_training:
      params = tf.trainable_variables()
      gradients = tf.gradients(self.loss, params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
        zip(clipped_gradients, params), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
