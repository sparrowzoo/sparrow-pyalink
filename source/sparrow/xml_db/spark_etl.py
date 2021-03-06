import os
import sys
from xml.dom.minidom import parse

from sparrow.xml_db.pyspark_config import Pyconfig


class SparkETL(Pyconfig):
    def __init__(self, xml_path=None, app_name="feature", spark_url="local", executor_memory="6g", executor_cores=8):
        self.SPARK_APP_NAME = app_name
        self.SPARK_URL = spark_url
        self.ENABLE_HIVE_SUPPORT = True
        self.SPARK_EXECUTOR_MEMORY = executor_memory
        self.SPARK_EXECUTOR_CORES = executor_cores
        self.SPARK_EXECUTOR_INSTANCES = executor_cores
        self.spark = self.create_spark_session()
        self.dict = {}
        os.chdir(sys.path[0])
        if xml_path != None :
            doc = parse(xml_path)
            select_list = doc.getElementsByTagName('select')
            for select in select_list:
                name = select.getAttribute('name')
                self.dict[name] = select.childNodes[0].nodeValue

    def getSession(self):
        return self.spark

    def load(self, name, args=0):
        sql = self.dict[name]
        print(sql)
        return self.spark.sql(sql.format(**args))  ##** args