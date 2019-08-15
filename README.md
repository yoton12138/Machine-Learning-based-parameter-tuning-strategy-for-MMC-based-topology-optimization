# Machine-Learning-based-parameter-tuning-strategy-for-MMC-based-topology-optimization
all code for this study in MMC-master.zip

MMC_cantilever和MMC_Lshape函数是分别是两个算例的求解函数，FUNCTION文件夹是子函数。

pso.m运行可以得到论文中的结果，对应不同算例需要改变pso.m中对应的求解函数，这里指MMC_cantilever.m和MMC_Lshape.m。

pkl文件是利用机器学习训练好的模型，后缀为ET的对应是cantilever算例的模型。后缀ET-L5的是Lshape算例的模型。
