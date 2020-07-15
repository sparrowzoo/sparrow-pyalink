from sparrow.strategy.abstract_model import AbstractModel
from sparrow.xml_db.etl import ETL


class LRModel(AbstractModel):
    def __init__(self):
        pass

    def run(self, args):
        etl = ETL("recommend_lr_feature")
        sample = etl.load("sample")
        user_feature = etl.load("user_feature")
        item_feature = etl.load("item_feature")

        result = predict(sample, user_feature, item_feature)
        save_result(result)
