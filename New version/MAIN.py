# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:17:17 2017

@author: vento
"""

import os
from skimage import io
import numpy as np
import OperatePicture as OP
import OperateDataBase as OD
import csv

##Essential vavriable 基础变量
#Standard size 标准大小
N = 100
#Gray threshold 灰度阈值
color = 100/255

#读取文件
reader = list(csv.reader(open('DataBase.csv', encoding = 'utf-8')))
del reader[0]
fileNames = os.listdir(r"./num/")
newFileNames = OD.NewFiles(fileNames, reader)
pic = OP.GetTrainPicture(newFileNames)
print(newFileNames)
OD.SaveToCSV(pic, newFileNames)
pic = OD.HeBing(reader, pic)

