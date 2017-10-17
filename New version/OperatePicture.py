# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:47:21 2017

@author: vento
"""

import os
from skimage import io
import numpy as np

##Essential vavriable 基础变量
#Standard size 标准大小
N = 100
#Gray threshold 灰度阈值
color = 100/255

def JudgeEdge(img, length, flag, size):
    for i in range(length):
        #Cow or Column 判断是行是列
        if flag == 0:
            #Positive sequence 正序判断该行是否有手写数字
            line1 = img[i, img[i,:]<color]
            #Negative sequence 倒序判断该行是否有手写数字
            line2 = img[length-1-i, img[length-1-i,:]<color]
        else:
            line1 = img[img[:,i]<color, i]
            line2 = img[img[:,length-1-i]<color,length-1-i]
        #If edge, recode serial number 若有手写数字，即到达边界，记录下行
        if len(line1)>=1 and size[0]==-1:
            size[0] = i
        if len(line2)>=1 and size[1]==-1:
            size[1] = length-1-i
        #If get the both of edge, break 若上下边界都得到，则跳出
        if size[0]!=-1 and size[1]!=-1:
            break
    return size

#Cut the Picture 切割图象
def CutPicture(img):
    #初始化新大小
    size = []
    #图片的行数
    length = len(img)
    #图片的列数
    width = len(img[0,:])
    #计算新大小
    size.append(JudgeEdge(img, length, 0, [-1, -1]))
    size.append(JudgeEdge(img, width, 1, [-1, -1]))
    size = np.array(size).reshape(4)
    return img[size[0]:size[1]+1, size[2]:size[3]+1]

def StretchPicture(img):
    newImg1 = np.ones(N*len(img)).reshape(len(img), N)
    newImg2 = np.ones(N**2).reshape(N, N)
    #对每一行进行拉伸/压缩
    #每一行拉伸/压缩的步长
    temp1 = len(img[0])/100
    #每一列拉伸/压缩的步长
    temp2 = len(img)/100
    #对每一行进行操作
    for i in range(len(img)):
        for j in range(N):
            newImg1[i, j] = img[i, int(np.floor(j*temp1))]
    #对每一列进行操作
    for i in range(N):
        for j in range(N):
            newImg2[i, j] = newImg1[int(np.floor(j*temp2)), j]
    return newImg2

#Read and save train picture 读取训练图片并保存
def GetTrainPicture(files):
    Picture = np.zeros([len(files), N**2+1])
    #loop all pictures 循环所有图片文件
    for i, item in enumerate(files):
        #Read the picture and turn RGB to grey读取这个图片并转为灰度值
        img = io.imread('./num/'+item, as_grey = True)
        #Clear the noise清除噪音
        img[img>color] = 1
        #Cut the picture and get the picture of handwritten number
        #将图片进行切割，得到有手写数字的的图像
        img = CutPicture(img)
        #Stretch the picture and get the standard size 100x100
        #将图片进行拉伸，得到标准大小100x100
        img = StretchPicture(img).reshape(N**2)
        #Save the picture to the matrix 将图片存入矩阵
        Picture[i, 0:N**2] = img
        #Save picture's name to the matrix 将图片的名字存入矩阵
        Picture[i, N**2] = int(item[0])
    return Picture