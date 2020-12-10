import os

import tensorflow as tf

# PYSPARK_PYTHON=D:\Users\MC\Anaconda3\envs\tensorflow\python.exe
os.environ['PYSPARK_PYTHON']="D:\\Users\\MC\\Anaconda3\\envs\\tensorflow\\python.exe"

sess=tf.Session()
#特征数据
features = {
    'sex': ['male', 'male', 'female', 'female', 'da'],
    'age': [1, 2, 3, 4, 1],
    'sale': [1.2, 2.3, 1.2, 1.5, 2.2]
}
#特征列
sex_column = tf.feature_column.categorical_column_with_vocabulary_list('sex', ['male', 'female'])
sex_column = tf.feature_column.indicator_column(sex_column)
age_column = tf.feature_column.categorical_column_with_identity('age', num_buckets=3, default_value=0)
age_column = tf.feature_column.bucketized_column(age_column)
sale_column = tf.feature_column.numeric_column("sale", default_value=0.0)

#组合特征列
columns = [sale_column,sex_column,age_column]
#输入层（数据，特征列）
inputs = tf.feature_column.input_layer(features, columns)

#初始化并运行
init = tf.global_variables_initializer()
sess.run(tf.tables_initializer())
sess.run(init)
v=sess.run(inputs)
print(v)