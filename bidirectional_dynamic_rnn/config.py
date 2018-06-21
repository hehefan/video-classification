import tensorflow as tf

tf.app.flags.DEFINE_integer("num_epochs", 200, "Number of epochs.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers.")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("min_learning_rate", 1e-4, "Minimal learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "Dropout between the last state and the full connection.")
tf.app.flags.DEFINE_float("input_keep_prob", 1.0, "Input dropout of cells.")
tf.app.flags.DEFINE_float("output_keep_prob", 1.0, "Output dropout of cells.")
tf.app.flags.DEFINE_float("state_keep_prob", 1.0, "State dropout of cells.")

tf.app.flags.DEFINE_integer("cell_size", 512, "Size of RNN cell.")
tf.app.flags.DEFINE_boolean("use_lstm", True, "GRU or LSTM.")

tf.app.flags.DEFINE_integer("feature_size", 2048, "Size of frame feature.")
tf.app.flags.DEFINE_integer("max_video_length", 75, "Maximal length of video.")

tf.app.flags.DEFINE_integer("num_classes", 101, "Number of classes.")

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 400, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory for saving checkpoints.")

FLAGS = tf.app.flags.FLAGS
