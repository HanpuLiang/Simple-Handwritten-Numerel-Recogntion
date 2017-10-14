# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:17:17 2017

@author: vento
"""

import os
from skimage import io
import numpy as np
import OperatePicture as OP

##Essential vavriable 基础变量
#Standard size 标准大小
N = 100
#Gray threshold 灰度阈值
color = 100/255

filenames = os.listdir(r"./num/")
pic = OP.GetTrainPicture(filenames)