import tensorflow as tf

tf.app.flags.DEFINE_integer("num_epochs", 50, "Number of epochs.")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("min_learning_rate", 1e-5, "Minimal learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "Dropout.")

tf.app.flags.DEFINE_integer("size", 512, "Size of RNN cell.")
tf.app.flags.DEFINE_boolean("use_lstm", False, "GRU or LSTM.")

tf.app.flags.DEFINE_integer("feature_size", 1024, "Size of frame feature.")
tf.app.flags.DEFINE_integer("max_video_length", 300, "Maximal length of video.")

tf.app.flags.DEFINE_integer("num_classes", 2, "Number of classes.")

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoint_mean", "Directory for saving checkpoints.")

FLAGS = tf.app.flags.FLAGS
