在提交spark的时候，我们往往python工程是多个python文件，彼此之间有调用关系。
那如何提交python工程呢？
./bin/spark-submit –py-files project_name.zip main.py即可

project_name是你将你所有需要用到的python文件打包成一个zip文件
main是你的python文件的main函数所在的py文件。


zip -r "sparrow" ./* -x "data/" -x ".git" -x ".pyc"
