import os

import pandas as pd


from sparrow.xml_db.spark_etl import SparkETL

# PYSPARK_PYTHON=D:\Users\MC\Anaconda3\envs\tensorflow\python.exe
os.environ['PYSPARK_PYTHON']="D:\\Users\\MC\\Anaconda3\\envs\\tensorflow\\python.exe"


COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income_bracket"
]
print(COLUMNS)
adult = pd.read_csv('../adult.data', names=COLUMNS, engine='python')
print(adult)
print(adult.columns)
print(adult.head(1))
print(type(adult))



etl = SparkETL(app_name="feature")
# 无需设置表结构

spark_train = etl.getSession().createDataFrame(adult)

spark_train.write.mode("overwrite").saveAsTable("adult_data")
etl.getSession().sql("select * from adult_data").show()
etl.getSession().sql("desc adult_data").show()



