from sparrow.strategy.abstract_model import AbstractModel
from sparrow.xml_db.spark_etl import SparkETL


class LRModel(AbstractModel):
    def __init__(self):
        pass

    def run(self, args):
        etl = SparkETL("sparrow/xml_db/recommend_lr_feature.xml", "recommend-lr")
        etl.getSession().sql("show databases").show()
        etl.getSession().sql("show tables").show()

        # sample = etl.load("sample",args)
        # user_feature = etl.load("user_feature", 1)
        # item_feature = etl.load("item_feature", 1)

        # result = predict(sample, user_feature, item_feature)
        # save_result(result)