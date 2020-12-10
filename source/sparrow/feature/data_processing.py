from pyspark.ml.feature import Binarizer, Bucketizer, QuantileDiscretizer, OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
from functools import reduce


def binarization_by_threshold(dataFrame, threshold, inputCol):
    # 对连续值根据阈值threshold二值化
    binarizer = Binarizer(threshold=threshold, inputCol=inputCol, outputCol='%s_binarized' % (inputCol))
    binarizedDataFrame = binarizer.transform(dataFrame)
    print('Binarizer output with Threshold = %f' % binarizer.getThreshold())
    return binarizedDataFrame

def bucketizer_splits(dataFrame, inputCol, splits=[-float('inf'), -0.5, 0.0, 0.5, float('inf')]):
    # 按给定边界分桶离散化——按边界分桶
    bucketizer = Bucketizer(splits=splits, inputCol=inputCol,
                            outputCol='%s_bucketizer' % (inputCol))  # splits指定分桶边界
    bucketedData = bucketizer.transform(dataFrame)
    print('Bucketizer output with %d buckets' % (len(bucketizer.getSplits()) - 1))
    return bucketedData

def quantile_discretizer(dataFrame, inputCol, numBuckets=4):
    # 按分位数分桶离散化——分位数离散化
    discretizer = QuantileDiscretizer(numBuckets=numBuckets, inputCol=inputCol,
                                      outputCol='%s_bucketizer' % (inputCol))  # numBuckets指定分桶数
    bucketedData = discretizer.fit(dataFrame).transform(dataFrame)
    return bucketedData

def log_feature(dataFrame, inputCol):
    def f(x):
        import numpy as np
        # 可以自己设定一些变换函数比如取log或者其他的方式 让长尾数据变成分布较合理 然后归一化
        return float(np.log(x + 1)) if x > -1 else -1.0

    udf_log = udf(f, FloatType())
    return dataFrame.withColumn("%s_log" % inputCol, udf_log(dataFrame[inputCol]))

def max_abs_scaler(dataFrame, inputCol, scaler = [0,1]):
    #将数据归一化到scaler[a,b]区间
    #（1）首先找到样本数据Y的最小值Min及最大值Max
    #（2）计算系数为：k=（b-a)/(Max-Min)
    #（3）得到归一化到[a,b]区间的数据：norY=a+k(Y-Min)
    describe = dataFrame.describe() #dataFrame.describe().fillna(0)
    maxValue = float(describe.filter(describe["summary"]=="max").select(inputCol).first()[0])#summary:count, mean, stddev, min, max
    minValue = float(describe.filter(describe["summary"]=="min").select(inputCol).first()[0])#summary:count, mean, stddev, min, max
    k = (scaler[1] - scaler[0])/(maxValue - minValue)
    def f(x):
        return scaler[0] + k * (x-minValue)
    udf_scaler = udf(f, FloatType())
    return dataFrame.withColumn("%s_scaler" % inputCol, udf_scaler(dataFrame[inputCol]))

def label_encode(dataFrame, inputCol):
    #label编码
    #string类型转化成double类型
    stringIndexer=StringIndexer(inputCol=inputCol,outputCol='%s_label' % (inputCol))
    model=stringIndexer.fit(dataFrame)
    indexedData=model.transform(dataFrame)
    return indexedData

def one_hot_encode(dataFrame, inputCol, idCol):
    labelEncodeDataFrame = label_encode(dataFrame,inputCol)

    one_hot_len = labelEncodeDataFrame.select(inputCol).distinct().count()
    def one_hot(x):
        one_hot_list = [0]*one_hot_len
        one_hot_list[x] = 1
        return one_hot_list
    tmp = labelEncodeDataFrame.select(idCol, '%s_label' % (inputCol)).rdd.\
                map(lambda x:[int(x[0])] + one_hot(int(x[1]))). \
                toDF("%s:int"%idCol + reduce(lambda x, y: x + y, [',%s_%d:int' % (inputCol, i) for i in range(one_hot_len)]))

                # toDF("id:int" + reduce(lambda x, y: x + y, [',%s_%d:int' % (inputCol, i) for i in range(one_hot_len)]))
    return dataFrame.join(tmp, [idCol], "left")

if __name__ == "__main__":
    # '使用举例:'
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName('Example').getOrCreate()
    data = spark.createDataFrame([(0,1.1),(1,8.5),(2,5.2),(3,4.2),(4,3.2),(4,6.2)],['id','feature'])
    data.show()
    print(data.describe())

    # #dataframe转rdd，利用rdd对每一行进行函数处理，然后再转成dataframe，合并原来的dataframe
    # import numpy as np
    # b = data.select(data['id'],data['feature']).\
    #         rdd.map(lambda x: [x[0],float(np.log(x[1]+1) if x[1] > 5 else -1)]).\
    #         toDF("id:int,feature:double")
    # data.join(b, data.id==b.id).take(3)
    #
    # binarization_by_threshold(data, 5.0, 'feature').show()
    # bucketizer_splits(data, 'feature', splits=[-float('inf'),2,6,8,float('inf')]).show()
    # quantile_discretizer(data, 'feature', 5).show()
    # print('log_feature:',log_feature(data, 'feature').show())
    # max_abs_scaler(data, 'id').show()
    # label_encode(data,'feature').show()
    # one_hot_encode(data, 'feature', 'id').show()
    data = quantile_discretizer(data, 'feature', 5)
    one_hot_encode(data, 'feature_bucketizer', 'id').show()
