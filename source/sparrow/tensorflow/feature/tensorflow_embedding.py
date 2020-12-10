import tensorflow as tf
sess = tf.Session()
# 特征数据
features = {
    'department': ['sport', 'travelling','drawing', 'gardening'],
}
# 特征列
department = tf.feature_column.categorical_column_with_hash_bucket('department', 5, dtype=tf.string)
department_indicator = tf.feature_column.indicator_column(department)
department_one_hot = tf.feature_column.input_layer(features, [department_indicator])

with tf.Session() as session:
        print(session.run([department_one_hot]))
"—————1—————"
# print columns
columns = tf.feature_column.embedding_column(department, dimension=4)
# 输入层（数据，特征列）
inputs = tf.feature_column.input_layer(features, columns)
# 初始化并运行
init = tf.global_variables_initializer()
sess.run(tf.tables_initializer())
sess.run(init)

v = sess.run(inputs)
print(v)