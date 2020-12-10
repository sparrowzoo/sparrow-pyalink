import os

import pandas as pd

from sparrow.xml_db.spark_etl import SparkETL

# PYSPARK_PYTHON=D:\Users\MC\Anaconda3\envs\tensorflow\python.exe
os.environ['PYSPARK_PYTHON']="D:\\Users\\MC\\Anaconda3\\envs\\tensorflow\\python.exe"

train = pd.read_csv('./flight_date.csv')
print(train)
print(train.columns)
print(train.head(1))
print(type(train))

etl = SparkETL(app_name="feature")

spark_train = etl.getSession().createDataFrame(train)
# 设置表的字段名
columns = ''
for i in train.columns:
    columns += '%s String,' % i
# 拼写SQL语句
sql_str = 'create table if not exists flight_date ({})'.format(columns[:-1])  # 最后一个逗号需要去掉，否则报错
etl.getSession().sql(sql_str)  # 执行SQL
etl.getSession().sql("desc flight_date").show()  # 执行SQL

spark_train.write.mode("overwrite").saveAsTable("flight_date")
etl.getSession().sql("select * from flight_date").show()
