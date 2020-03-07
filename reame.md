linux下运行命令
```
vi ~/.pip/pip.conf (.pip)
```
然后写入如下内容并保存
windows:
```
%HOMEPATH%\pip\pip.ini (pips)
```
%HOMEPATH% 用户的家目录，非pip 安装目录
```
 [global]
 trusted-host =  mirrors.aliyun.com
 index-url = https://mirrors.aliyun.com/pypi/simple
```
Python Package Index (PyPI) 
https://pypi.org/project/dill/0.2.9/
pip install dill==0.2.9
pip install pyalink-1.1.0-py3-none-any.whl

https://docs.microsoft.com/zh-cn/windows/python/

window 查看安装目录
where pip
linux 查看安装目录
whereis python2

```
could not install packages due to an EnvironmentError:[WinError 5 ]
```
加--user

```
ERROR: Cannot uninstall 'PyYAML'. It is a distutils installed project and thus we cannot accurately
```
加  --ignore-installed PyYAML

最终安装脚本
---
```
$ pip install pyalink-1.1.0-py3-none-any.whl --ignore-installed PyYAML --user
```
源码安装
---
`flink python 不支持3.8`
- https://ci.apache.org/projects/flink/flink-docs-release-1.10/tutorials/python_table_api.html
- https://ci.apache.org/projects/flink/flink-docs-release-1.10/flinkDev/building.html#build-pyflink
```bash
python3.7 -m pip install *.tar.gz
```


在执行 python -m pip install apache-flink 时遇到错误：

Command "python setup.py egg_info" failed with error code 1
解决方法是更新 setuptools 和 pip：
```
pip3 install --upgrade setuptools
python -m pip install --upgrade pip
```
REPL
Read Eval Print Loop
读取，执行，输出，循环
在REPL环境中，你输入一句话，他就读取，执行，输出一个结果，所以也称为 交互式提示模式


UDF （User-Defined Function）(用户定义函数)


https://ci.apache.org/projects/flink/flink-docs-release-1.10/tutorials/python_table_api.html
 