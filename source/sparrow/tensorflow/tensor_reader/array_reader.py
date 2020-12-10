import tensorflow as tf
import numpy as np
from tensorflow.contrib.eager.python import tfe

tf.enable_eager_execution()

# 创建dataset
dataset = tf.data.Dataset.from_tensor_slices(
                                             np.array([1.0, 2.0, 3.0, 4.0, 5.0])
                                             )
# # 实例化了一个Iterator
# iterator = dataset.make_one_shot_iterator()
# # 从iterator里取出一个元素
# one_element = iterator.get_next()

for one_element in tfe.Iterator(dataset):
    print(one_element)