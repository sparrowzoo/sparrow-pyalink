from functools import reduce

from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

from sparrow.feature.data_processing import one_hot_encode
from sparrow.strategy.abstract_model import AbstractModel
from sparrow.xml_db.spark_etl import SparkETL


class LRModel(AbstractModel):

    def __init__(self):
        pass

    def run(self, args):
        etl = SparkETL("sparrow/xml_db/recommend_lr_feature.xml", "feature")
        # etl.getSession().sql("show databases").show()
        # etl.getSession().sql("show tables").show()
        dataframe = etl.getSession().sql("select * from label_onehot_test")
        dataframe.show()
        dataframe = one_hot_encode(dataframe,"label","id")
        # etl.getSession().sql("drop table python_feature_table")
        dataframe.show()
        dataframe.write.mode("overwrite").saveAsTable("python_feature_table")
        result= etl.getSession().sql("select `label-onehot` from python_feature_table")


        # sample = etl.load("sample",args)
        # user_feature = etl.load("user_feature", 1)
        # item_feature = etl.load("item_feature", 1)

        # result = predict(sample, user_feature, item_feature)
        # save_result(result)