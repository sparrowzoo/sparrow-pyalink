import os

import tensorflow.contrib.eager as tfe
import tensorflow as tf
import numpy as np
tfe.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
dataset = tf.data.Dataset.from_tensor_slices(
                                            np.array([
                                                     [[1, 2, 3,],
                                                      [4, 5, 6]],
                                                     [[7, 8, 9,],
                                                      [10,11,12]]
                                                      ])
                                            )
print("dataset:",dataset)
for one_element in tfe.Iterator(dataset):
    print(one_element)
