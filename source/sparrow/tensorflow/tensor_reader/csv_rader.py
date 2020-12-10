import tensorflow as tf

# 定义输入样本格式
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]
_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]


def print_tensor(dataset):
    # 实例化了一个Iterator
    iterator = dataset.make_one_shot_iterator()  # <tensorflow.python.data.ops.iterator_ops.Iterator object at 0x000002016B501CC0>
    # 从iterator里取出一个元素
    # <tf.Tensor 'IteratorGetNext:0' shape=() dtype=float64>
    # 从iterator里取出一个元素
    one_element = iterator.get_next()  # <tf.Tensor 'IteratorGetNext:0' shape=() dtype=float64>
    print(type(one_element))
    count=0
    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(one_element))
                count+=1
                # if count==2:
                #     break
            except tf.errors.OutOfRangeError:
                break
    print(count)

def fetch_dataset_from_csv(data_file, _CSV_COLUMNS, _CSV_COLUMN_DEFAULTS, num_parallel_calls):
    assert tf.gfile.Exists(data_file), "{0} not found.".format(data_file)

    def parse_csv(line):
        print(type(line))
        print("Parsing", data_file)
        # tf.decode_csv会把csv文件转换成Tensor,一列一个。其中record_defaults用于指明每一列的缺失值用什么填充。
        columns = tf.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        # tf.equal(x, y) 返回一个bool类型Tensor， 表示x == y, element-wise
        return features, tf.equal(labels, '>50K')

    dataset = tf.data.TextLineDataset(data_file).map(parse_csv, num_parallel_calls=num_parallel_calls)
    return dataset


train_file = '../adult.data'
test_file = '../adult.test'

for i in range(10):
    train_dataset = fetch_dataset_from_csv(train_file, _CSV_COLUMNS, _CSV_COLUMN_DEFAULTS, 5)
    train_dataset = train_dataset.repeat(1)
    train_dataset = train_dataset.batch(10000)
    print_tensor(train_dataset)
    iterator = train_dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
