- 提交spark  
1. 在提交spark的时候，我们往往python工程是多个python文件，彼此之间有调用关系。
那如何提交python工程呢？
./bin/spark-submit –py-files project_name.zip main.py即可
2. project_name是你将你所有需要用到的python文件打包成一个zip文件
main是你的python文件的main函数所在的py文件。

- 打包命令
```
-x   exclude the following names 排除某文件
```
```
zip -r "sparrow-src" ./source/* -x "data/" -x ".git" -x ".pyc"
```
http://spark.apache.org/docs/latest/submitting-applications.html#master-urls
```
spark-submit --master local --py-files sparrow-src.zip --executor-memory 2G source/main.py
spark-submit --master local --py-files sparrow-src.zip --executor-memory 2G main.py

```
spark-submit online command line
```
/opt/spark/spark/bin/spark-submit --name new_corss_cover --queue etl-sc --master yarn --deploy-mode cluster --archives hdfs://ns1/user/ruban/deploy/pyspark_env/py3_ml_env/search_recommendation.zip --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./search_recommendation.zip/search_recommendation/bin/python --conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=./search_recommendation.zip/search_recommendation/bin/python --executor-memory 6g --num-executors 8 --executor-cores 8 --driver-memory 6g --files $SPARK_HOME/conf/hive-site.xml --py-files meicai_index_cross_main.zip test_xml_main.py 
```

# spark on yarn
##两种模式：
    client模式（测试）
    cluster模式（生产常用）
    
##区别：
    client模式的driver是运行在客户端。
    cluster模式的driver运行在NodeManager的MRApplicationMaster中。
 
 
### client模式
启动：
```
    spark-submit --master yarn
    spark-submit --master yarn-client
    spark-submit --master yarn --deploy-mode client
```
 
 
执行流程：

    1. 创建sparkContext对象时，客户端先在本地启动driver，然后客户端向ResourceManager节点申请启
        动应用程序的ApplicationMaster。
    2. ResourceManager收到请求，找到满足资源条件要求的NodeManager启动第一个Container,然
        后要求该NodeManager在Container内启动ApplicationMaster。
    3. ApplicationMaster启动成功则向ResourceManager申请资源，ResourceManager收到请求给
        ApplicationMaster返回一批满足资源条件的NodeManager列表。
    4. Applicatio拿到NodeManager列表则到这些节点启动container，并在container内启动executor,executor启动成功则会向driver注册自己。
    5. executor注册成功，则driver发送task到executor,一个executor可以运行一个或多个task。
    6. executor接收到task,首先DAGScheduler按RDD的宽窄依赖关系切割job划分stage,然后将stage以TaskSet的方式提交给TaskScheduler。
    7. TaskScheduler遍历TaskSet将一个个task发送到executor的线程池中执行。
    8. driver会监控所有task执行的整个流程，并将执行完的结果回收。
 
弊端：

    由于client模式下driver运行在客户端，当应用程序很多且driver和worker有大量通信的时候，
        会急剧增加driver和executor之间的网络IO。
    并且大量的dirver的运行会对客户端的资源造成巨大的压力。
 
 
### cluster模式
启动：

    spark-submit --master yarn-cluster
    spark-submit --master yarn --deploy-mode cluster
  
执行流程：

    1、创建sparkContext对象时，客户端向ResourceManager节点申请启动应用程序的
        ApplicationMaster。
    2、ResourceManager收到请求，找到满足资源条件要求的NodeManager启动第一个Container,然
        后要求该NodeManager在Container内启动ApplicationMaster。
    3、ApplicationMaster启动成功则向ResourceManager申请资源，ResourceManager收到请求给
        ApplicationMaster返回一批满足资源条件的NodeManager列表。
    4、Applicatio拿到NodeManager列表则到这些节点启动container，并在container内启动executor,executor启动成功则会向driver注册自己。
    5、executor注册成功，则driver发送task到executor,一个executor可以运行一个或多个task。
    6、executor接收到task,首先DAGScheduler按RDD的宽窄依赖关系切割job划分stage,然后将stage以TaskSet的方式提交给TaskScheduler。
    7、TaskScheduler遍历TaskSet将一个个task发送到executor的线程池中执行。
    8、driver会监控所有task执行的整个流程，并将执行完的结果回收。
 
 
    cluster模式下driver分散运行在集群节点，有效避免了client的问题。生产用的就是cluster模式。
 
## 参数
```
 /opt/apps/spark-1.5.1-hadoop2.4/bin/spark-submit \
    --class org.apache.spark.examples.SparkPi \        #作业类名
    --master yarn-clinet \                       #spark模式
    --driver-memory 4g \                    #每一个driver的内存
    --num-executors 1 \
    --executor-memory 2g \                    #每一个executor的内存
    --executor-cores 1 \                        #每一个executor占用的core数量
    --queue thequeue \                            #作业执行的队列
    /opt/apps/spark-1.5.1-hadoop2.4/lib/spark-examples-1.5.1-hadoop2.4.0.jar \            #jar包
100 #传入类中所需要的参数
```
注：

    提交作业时spark会将lib下的spark-assembly*.jar包分发到yarn的am container中，这十分耗费资源。
    故而我们可以将该jar包放在一个yarn可以访问到的目录中，具体做法如下：
        vi spark-default.conf  
             spark.yarn.jars  hdfs://Linux001:8020/somepath/spark-assembly*.jar
        然后将jar包上传至hdfs://Linux001/somepath/即可。
    这是一个调优点哦
## spark on yarn的个性化启动
    /spark-shell --master yarn --jars mysql驱动包绝对路径
    简化1：省去--jars
        vi spark-defaults.conf  
            spark.executor.extraClassPath  mysql驱动包绝对路径
            spark.driver.extraClassPath  mysql驱动包绝对路径
        当某个jar包或类找不到的时候，可以参考这一点
    简化2：省去--master
        vi spark-defaults.conf
            #spark.master local[3]  取消注释 local[3]  改为yarn
注意：
    这种对于spark-defaults.conf的修改，会作用于其他spark模式，容易产生问题，怎么办？
解决：
    拷贝spark-defaults.conf.template,制作自己的个性化配置文件
    启动时添加：
        --properties-file ${配置文件}  即可
 
 
## 重要的属性
spark.port.maxRetries
    同时运行的最大作业数目，默认16,肯定不够，建议调大
 
spark.yarn.maxAppAttempts 
    作业最大重试次数，它不应该大于yarn配置中设置的全局最大重试数。默认是等于的。

