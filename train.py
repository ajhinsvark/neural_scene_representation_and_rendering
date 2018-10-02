import tensorflow as tf
from utils.data_reader import DataReader

root_path = 'data'
data_reader = DataReader(dataset='shepard_metzler_5_parts', context_size=5, root=root_path)
data = data_reader.read(batch_size=12)

with tf.train.SingularMonitoredSession() as sess:
    d = sess.run(data)