# StyleGAN_v1 文档结构

## (1). StyleGAN.py / train_net.py 文件

- save_sample()
  1. tracker参数

     > from nativeUtiles.tracker import LossTracker

  2. logger参数

     > import logging 这是一个第三方库

- train()
  1. Model()对象
  2. lod2batch对象

  > 来自 import module.lod_driver as lod_driver，记录各类参数

  3. 这个函数主要被train_net.py文件调用

     

- train_net.py

  1. 用于设置参数
  2. 启动train()函数

## (2). dlutils 文件夹

> deep learning utils 的简称，是作者自己做的一个包，用于模型的参数测试和输出

## (3).dataUtils 文件夹

- 实现了将tf中的数据格式tfrecords，转换为pytorch的数据格式。
- 具体实现是通过**pybind**做了一个c++库到python的迁移
- 再实现了一个将tfrecord中的zip数据集转换到pytorch的第三方库:**dareblopy** [链接](https://github.com/podgorskiy/DareBlopy)。

## (4).nativeUtils 文件夹

- checkpointer.py

  1.通过字典保持需要存储的对象

  2.主要包括

  ​	 save(),

  ​	 load(), 

  ​	 tag_last_checkpoint() 即标记训练时最后一个

## (5). module文件夹

* logdriver.py
  1. 类似一个驱动文件，将配置中的参数记录在一个类	
  2. 并随着分辨率(lod)的增长而通过内置方法改变: set_epoch()
* losses.py
  1. 默认用的是逻辑回归,其实D+gp 
  2. 总的包括 kl , BCE, LR
  3. 实现逻辑回归用的是 F.softplus(), 假值带负号输入

## (6).test文件夹

- tests.py

  ​	测试参数文件

- Sample-blob.py

  ​	测试压缩部分层的数值减少style-gan水滴的情况

- test_v1-v5.py

  ​	测试预训练模型下不同的结果(和官方一致)

  

