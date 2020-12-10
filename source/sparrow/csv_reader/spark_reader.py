from sparrow.xml_db.spark_etl import SparkETL

etl = SparkETL(app_name="feature")
flightDate=etl.getSession().read.option("inferSchema","true").option("header","true").csv("./flight_date.csv")
flightDate.sort("count").explain()
