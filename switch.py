from ET import *
import os
import numpy as np
execution_path = os.getcwd()
image_path = os.path.join(execution_path, "L")
files1 = os.listdir(image_path)#列出当前目录下所有的文件
for filename in files1:
    portion = os.path.splitext(filename)#分离文件名字和后缀
    print(portion)
    if portion[1] ==".jpg":#根据后缀来修改,如无后缀则空 
        newname = portion[0]+".png"#要改的新后缀
        os.chdir(image_path)#切换文件路径,如无路径则要新建或者路径同上,做好备份
        os.rename(filename,newname)