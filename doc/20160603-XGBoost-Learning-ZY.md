## XGBoost Learning

+ author: zhouyongsdzh@foxmail.com
+ date: 20160503

### 0. 目录

### 1. 写在前面

### 2. XGBoost入门

#### 2.1. 准备工作

+ XGBoost相关地址
	+ github: https://github.com/dmlc/xgboost.git
	+ 文档：
+ 安装

	+ 第一步：安装XGBoost
	+ 第二步：安装语言包

	XGBoost编译需要GCC版本在4.6及4.6+，安装过程参考[官方Installion文档](https://xgboost.readthedocs.io/en/latest/build.html#)

	1.1 在Ubuntu上安装步骤：

	```
	git clone --recursive https://github.com/dmlc/xgboost  
	# 因为使用submodules管理第三方package, 使用 --recursive可把third party下载下来
	cd xgboost
	make -j4		# 编译
	```

	>在Mac OS上安装步骤：
	>
	>```
	git clone --recursive https://github.com/dmlc/xgboost
	cd xgboost
	cp make/minimum.mk ./config.mk
	make -j4
	```
	>> 此安装是非多线程安装，因为OS 默认的clang，没有open-mp。
	
	1.2. 更高级的安装是Customized Building
	
	通过修改make/config.mk 使其工作在HDFS／Amazon S3等分布式文件系统中。
	
	2.1. Python Package Installation
	
	进入xgboost/python-package目录：
	
	```
	cd python-package; python setup.py install
	# 报错
	zhouyong@ubuntu:~/myhome/2016-Planning/ML-Model/xgboost-learning/python-package$ python setup.py installTraceback (most recent call last):  File "setup.py", line 6, in <module>    from setuptools import setup, find_packagesImportError: No module named setuptools
	# 提示我缺少setuptools模块, 下面安装之：
	sudo apt-get install python-setuptools
	## 安装setuptools时，再次报错：
	E: Could not get lock /var/lib/dpkg/lock - open (11: Resource temporarily unavailable)
E: Unable to lock the administration directory (/var/lib/dpkg/), is another process using it
	## 提示“资源被锁不可用”，解决办法参考：http://www.2cto.com/os/201305/213648.html
	sudo rm /var/cache/apt/archives/lock
 	sudo rm /var/lib/dpkg/lock
 	.
 	再次安装：sudo apt-get install python-setuptools 正常
 	.
 	安装python setup.py install时，再次提出错误
 	解决办法：ImportError: No module named setuptools 解决方案 
	shell中输入：
	wget http://pypi.python.org/packages/source/s/setuptools/setuptools-0.6c11.tar.gz
	tar zxvf setuptools-0.6c11.tar.gz
	cd setuptools-0.6c11
	python setup.py build
	python setup.py install
	.
	再次安装 xgboost python package, 安装过程中再次出现错误：
	Download error: unknown url type: https -- Some packages may not be found!   
	...
	AttributeError: 'NoneType' object has no attribute 'clone'
	提示错误为：https协议没有安装。解决方案：安装openssl-devel，然后重新编译安装python
	apt-get install openssl-dev
	sudo apt-get install libssl-dev
	再次安装xgboost pyhon package, 报错：ImportError: No module named numpy.distutils.core
	缺少工具：numpy, 安装之（源码安装, 安装之前需要安装上游以来并且配置，numpy才可用）
	# 官方安装方法 http://www.scipy.org/install.html，比较慢，还是手动安装吧
	# 可参考：http://blog.mimvp.com/2014/04/linux-install-numpy-and-scipy/ （保证上游以来能完整的安装）
	再次安装xgboost pyhon package, python setup.py install
	最后提示：
	Adding scipy 0.17.1 to easy-install.pth file	Installed /usr/local/lib/python2.7/site-packages/scipy-0.17.1-py2.7-linux-x86_64.egg	Searching for numpy==1.10.0	Best match: numpy 1.10.0	Adding numpy 1.10.0 to easy-install.pth file	Using /usr/local/lib/python2.7/site-packages	Finished processing dependencies for xgboost==0.4
	至此，安装完成done
	```
	
	> Note：安装过程一定手段确定python基础module是否已经安装成功，比如系统是否有openssl-dev组件（如果没有安装后，重新编译python），有没有setuptools工具，有没有numpy,scipy等，这些都需要在安装xgboost python package之前 需要安装的。
	
#### 2.2. XGBoost示例：Binary Classification

用XGBoost跑一个Demo，参考官方地址：https://github.com/dmlc/xgboost/tree/master/demo/binary_classification

核心过程

```
cd {path to xgboost}/demo/binary_classification
python mapfeat.py   # 获取<featureId, featureName>映射表, 其中第三列i表示（该值为二值数据）
python mknfold.py agaricus.txt 1  # 划分train与test （libsvm格式）
# 运行，mushroom.conf是参数配置文件，也可以在终端配置
../../xgboost mushroom.conf max_depth=10 num_round=10 save_period=2 nthread=4 model_dir="/home/zhouyong/myhome" 2> log.txt
# 预测 task=pred
../../xgboost mushroom.conf task=pred model_in=0003.model
# dump model
../../xgboost mushroom.conf task=dump model_in=0003.model name_dump=dump.raw.txt
../../xgboost mushroom.conf task=dump model_in=0003.model fmap=featmap.txt name_dump=dump.nice.txt
```

其中```featmap.txt```文件中，每一行结构：```<featureid> <featurename> <q or i or int>\n```	，示例：

```
0	cap-shape=bell	i1	cap-shape=conical	i2	cap-shape=convex	i3	cap-shape=flat	i4	cap-shape=knobbed	i5	cap-shape=sunken	i6	cap-surface=fibrous	i7	cap-surface=grooves	i8	cap-surface=scaly	i9	cap-surface=smooth	i10	cap-color=brown	i11	cap-color=buff	i12	cap-color=cinnamon	i13	cap-color=gray	i14	cap-color=green	i
```

映射字典 几点说明：

+ Feature id must be from 0 to number of features, in sorted order.
+ ```i``` means this feature is binary indicator feature
+ ```q``` means this feature is a quantitative value, such as age, time, can be missing
+ ```int``` means this feature is integer value (when int is hinted, the decision boundary will be integer)

训练数据调整观察：

一条样本：

```
0 3:1 7:1 20:1 21:1 24:1 34:1 37:1 40:1 51:1 54:1 55:1 65:1 69:1 77:1 86:1 88:1 92:1 95:1 102:1 106:1 118:1 126:1
```
改变featureID的顺序，交换```118:1 126:1```为```126:1 118:1```，

```
0 3:1 7:1 20:1 21:1 24:1 34:1 37:1 40:1 51:1 54:1 55:1 65:1 69:1 77:1 86:1 88:1 92:1 95:1 102:1 106:1 126:1 118:1
```

观察xgboost可否执行：**可以执行，说明xgboost不受libsvm中featureid的顺序影响**

> 打乱featureid顺序，与sorted featureId 训练模型结果相同。

### 2.3 XGBoost参数说明

参考地址：http://blog.csdn.net/zc02051126/article/details/46711047





	
