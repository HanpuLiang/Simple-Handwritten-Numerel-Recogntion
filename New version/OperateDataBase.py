# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 19:09:20 2017

@author: vento
"""

import os
import numpy as np
import OperatePicture as OP
import csv

##Essential vavriable 基础变量
#Standard size 标准大小
N = 100
#Gray threshold 灰度阈值
color = 100/255

def SaveToCSV(pic, fileNames):
   '''将pic与对应的dileNames存入CSV文件'''
    writer = csv.writer(open('DataBase.csv', 'a', newline = ''), dialect = 'excel')
    #将fileNames变为列表
    f = [item for item in fileNames]
    #每一行依次写入文件中
    for i in range(len(pic)):
        #将改行图片向量转为list
        item = pic[i].tolist()
        #将这个图片向量对应的名称f放入列表最后一个
        item.append(f[i])
        writer.writerow(item)
    

def NewFiles(fileNames, reader):
    '''判断是否有不同于数据库中的新文件加入'''
    #如果数据库中没有数据，则返回filenames
    if len(reader) == 0:
        return fileNames
    else:
        #从数据库中提取所有名称
        files = [item[10001] for item in reader]
        #需要加入的图片名称
        newFileNames = []
        for item in fileNames:
            #判断当前名称是否存在数据库中
            #如果不存在，则加入newFileNames
            if item not in files:
                newFileNames.append(item)
        return newFileNames

def HeBing(reader, pic):
    '''将两个矩阵reader与pic合并'''
    #两个矩阵的总行数
    l = len(reader) + len(pic)
    #初始化新的矩阵
    newPic = np.zeros(l*10001).reshape(l, 10001)
    #将reader最后的那个字符串名称去掉
    for item in reader:
        item.pop()
    #将reader转化为numpy的矩阵形式
    reader = np.array(reader)
    #新矩阵前半部分放reader，后半部分放pic
    if len(reader) != 0:
        newPic[0:len(reader), :] = reader
    newPic[len(reader):len(pic), :] = pic
    return newPic
