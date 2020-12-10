from pyspark import SparkConf
from pyspark.sql import SparkSession

''''
pyspark环境配置
_SPARK_APP_NAME spark环境名字
_SPARK_MASTER 使用yarn集群
_SPARK_EXECUTOR_MEMORY 该参数设置的是每个EXECUTOR分配的内存的数量
_SPARK_EXECUTOR_CORES 该参数为设置每个EXECUTOR能够使用的CPU core的数量。
_SPARK_EXECUTOR_INSTANCES yarn集群中，最多能够同时启动的EXECUTOR的实例个数
_ENABLE_HIVE_SUPPORT 是否支持hive
'''


class Pyconfig(object):
    SPARK_APP_NAME = None
    SPARK_MASTER = "local"
    SPARK_EXECUTOR_MEMORY = "4g"
    SPARK_EXECUTOR_CORES = 2
    SPARK_EXECUTOR_INSTANCES = 2
    ENABLE_HIVE_SUPPORT = True

    def create_spark_session(self):
        conf = SparkConf()
        config = (
            ("spark.app.name", self.SPARK_APP_NAME),
            ("spark.executor.memory", self.SPARK_EXECUTOR_MEMORY),
            ("spark.master", self.SPARK_MASTER),
            ("spark.executor.cores", self.SPARK_EXECUTOR_CORES),
            ("spark.executor.instances", self.SPARK_EXECUTOR_INSTANCES),
            ("spark.debug.maxToStringFields", "10000")
        )
        conf.setAll(config)
        if self.ENABLE_HIVE_SUPPORT:
            return SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
        else:
            return SparkSession.builder.config(conf=conf).getOrCreate()
