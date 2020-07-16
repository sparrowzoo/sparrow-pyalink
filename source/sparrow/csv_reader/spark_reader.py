from sparrow.xml_db.pyspark_config import Pyconfig


class SparkReader(Pyconfig):
    def __init__(self, xml_path="", app_name="recommend-api", spark_url="local", executor_memory="6g", executor_cores=8):
        self.SPARK_APP_NAME = app_name
        self.SPARK_URL = spark_url
        self.ENABLE_HIVE_SUPPORT = True
        self.SPARK_EXECUTOR_MEMORY = executor_memory
        self.SPARK_EXECUTOR_CORES = executor_cores
        self.SPARK_EXECUTOR_INSTANCES = executor_cores
        self.spark = self.create_spark_session()

flightDate=SparkReader().spark.read.option("inferSchema","true").option("header","true").csv("flight_date.csv")
flightDate.sort("count").explain()
