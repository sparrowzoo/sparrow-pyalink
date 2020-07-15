from sparrow.xml_db.etl import ETL

etl = ETL("recommend_feature")
order_count_7_day_feature = etl.load("order_count_7_day_feature")
order_count_180_day_feature = etl.load("order_count_180_day_feature")
