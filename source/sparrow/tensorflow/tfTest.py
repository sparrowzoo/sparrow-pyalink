import tensorflow.compat.v1 as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# The default path for saving event files is the same folder of this python file.
tf.app.flags.DEFINE_string('flag_log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs','Directory where event logs are written to.')

# Store all elements in FLAG structure!
FLAGS = tf.app.flags.FLAGS

# Defining some constant values
a = tf.constant(5.0, name="a")
b = tf.constant(10.0, name="b")

# Some basic operations
x = tf.add(a, b, name="add")
y = tf.divide(a, b, name="divide")


with tf.Session() as sess:
    log_path=os.path.expanduser(FLAGS.flag_log_dir)
    print(log_path)
    writer = tf.summary.FileWriter(log_path, sess.graph)
    print("output: ", sess.run([a,b,x,y]))
# Closing the writer.
writer.close()
print(tf.__version__)

