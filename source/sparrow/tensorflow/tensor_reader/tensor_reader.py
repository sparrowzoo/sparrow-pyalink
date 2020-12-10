import numpy as np
import tensorflow as tf

np.random.seed(0)
sample = np.random.sample((4, 2))
print(type(sample))
sample = [[0.60276338, 0.54488318],
          [0.5488135, 0.71518937],
          [0.4236548, 0.64589411],
          [0.5236548, 0.54589411],
          [0.6236548, 0.64589411],
          [0.73758721, 0.891773]]

def print_dataset(dataset):
    iter = dataset.make_one_shot_iterator()
    el = iter.get_next()
    with tf.Session() as sess:
        for i in range(1):
            value = sess.run(el)
            print(value)
# make a dataset from a numpy array

dataset = tf.data.Dataset.from_tensor_slices(sample)
dataset = dataset.shuffle(10)  # 将数据打乱，数值越大，混乱程度越大
dataset = dataset.repeat(3)  # 数据集重复了指定次数
dataset = dataset.batch(14)  # 按照顺序取出4行数据，最后一次输出可能小于batch
print_dataset(dataset)

# repeat()在batch操作输出完毕后再执行,若在之前，相当于先把整个数据集复制两次
# 为了配合输出次数，一般默认repeat()空

# create the iterator
# print_dataset(dataset)