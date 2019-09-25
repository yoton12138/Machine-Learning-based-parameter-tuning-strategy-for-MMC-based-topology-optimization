# Machine-Learning-based-parameter-tuning-strategy-for-MMC-based-topology-optimization
_all of the code for this study 

## 简单介绍
MMC_cantilever和MMC_Lshape函数是分别是两个算例的求解函数，FUNCTION文件夹是子函数。

pso.m运行可以得到论文中的结果，对应不同算例需要改变pso.m中对应的求解函数以及model，这里求解函数指MMC_cantilever.m和MMC_Lshape.m，model是图像分类模块的模型。

pkl文件是利用机器学习训练好的模型,用于图像识别，文件名末尾为ET的对应是cantilever算例的模型。末尾是ET-l的是Lshape算例的模型。

Lshape和cantilever-beam压缩包是机器学习的训练数据

* `完整运行代码需要安装scilit-learning 0.23.0版本`<br>
* `opencv 包含SIFT算法的全功能opencv，附件给出了安装方式`


## 各个代码的意思
### 需要更改路径的代码
* *PSO.m <br>*
  这是优化的主运行程序，不同的算例需要改变不同的求解函数，程序中表达为CostFuction的部分。
* *MMC188.m and MMC_Lshape.m* <br>
  分别对应两个算例的MMC优化迭代函数，需要更改图片保存路径。
* *switch.py* <br>
  将迭代的保存的jpg图片转化为png图片，以区别新旧的粒子种群迭代出的图像。需要更改图片的路径。
* *test.py* <br>
  对生成的图片进行识别，给定图片的标签，确定迭代的结构是否为可行解。需要对不同的例子更改不同的pkl模型文件，以及图片的路径。

### 不需要更改的代码
* *BasicKe.m*
* *Heaviside.m*:符号函数
* *tPhi.m*
* *tvoid.m*
* *variables.m*
* *subsolv.m*
* *mmasub.m*
* *afterProcessing.m* <br>
  程序自动运行结束后，会保存一个all.mat的文件，直接运行此代码可以将迭代过程中的粒子最优解pbest以及迭代顺序的全局最优解gbest等关键的程序结果提取出来。
## 程序结果
* 算例1 悬臂梁的优化结果
  ![算例1悬臂梁的优化结果](https://github.com/yoton12138/Machine-Learning-based-parameter-tuning-strategy-for-MMC-based-topology-optimization/blob/master/img/%E6%82%AC%E8%87%82%E6%A2%81%E4%BC%98%E5%8C%96%E7%BB%93%E6%9E%9C.png)
* 算例2 L型梁的优化结果 
  ![算例2L梁的优化结果](https://github.com/yoton12138/Machine-Learning-based-parameter-tuning-strategy-for-MMC-based-topology-optimization/blob/master/img/L%E5%9E%8B%E6%A2%81%E4%BC%98%E5%8C%96%E7%BB%93%E6%9E%9C.jpg)

