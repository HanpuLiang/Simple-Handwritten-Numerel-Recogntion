# Python徒手实现识别手写数字—简易图片数据库

## 写在前面

上一篇文章[Python徒手实现识别手写数字—图像的处理](http://www.jianshu.com/p/82387ae42587)中我们讲了图片的处理，将图片经过剪裁，拉伸等操作以后将每一个图片变成了1x10000大小的向量。但是如果只是这样的话，我们每一次运行的时候都需要将他们计算一遍，当图片特别多的时候会消耗大量的时间。

所以我们需要将这些向量存入一个文件当中，每次先看看图库中有没有新增的图片，如果有新增的图片，那么就将新增的图片变成1x10000向量再存入文件之中，然后从文件中读取全部图片向量即可。当图库中没有新增图片的时候，那么就直接调用文件中的图片向量进行计算就好。这样子算是节省了大量的时间。

所以本文就是从零开始建立一个这样的图片存储管理系统。

## 实现逻辑

### 第一次读入图片

我们的图库中拥有一大堆图片，每一张图片上面都是一个手写的数字，图片的名称为[数字内容]_[序号]。比如说一个图片的名称为2_3，代表这一张图片里面的数字是2，并且是“数字是2的第3张图片”。

存在一个csv文件作为我们的建议的图片数据库，名称为Data.csv。

首先我们读取图库中所有图片的名称，保存在fileNames中。然后读取Data.csv中所有数据。

提取出Data.csv的最后一列（一共10002列，第10001列说明该数字是什么数字，第10002列是图片的名称），也就是数据库中存储的所有图片的名称，存储在item中。

将新加入图库的图片名称保存在newFileNames中。如果Data.csv为空，那么就直接令newFileNames = fileNames。也就是说如果数据库中什么也没有，那么图库中所有图片都是新加入的。

如果Data.csv不为空，那么就将item里面的内容与fileNames的内容比较，如果出现了fileNames里面有的名称item中没有，那么就将这些名称放进newFileNames中。如果item里有的名称fileNames中没有，那就不管。

也就是说，我令我们的数据库只进不出。

现在我们得到了新加入图库的图片的名称newFileNames。

将newFileNames中的名称的图片带入上一文中函数GetTrainPicture进行处理，得到了一个nx10001的矩阵，每一行代表一个新加入的图片，前10000列是图片向量，第10001列是该图片的数字，保存在pic中。

将这些图片压入到数据库的后面。

读取之前数据库原有的图片向量，并与pic合并，得到目前拥有的所有的训练图片向量pic。

```flow
st=>start: 图库图片名称fileNames
cd1=>condition: 数据库为空？
op1=>operation: 全部导入newFileNames
op2=>operation: 数据库图片名称item
op3=>operation: 比较得到newFileNames
op4=>operation: 带入GetTrainPicture函数
op5=>operation: 得到矩阵pic
op6=>operation: 与数据库中原有数据合并
ed=>end: 压入数据库

st->cd1
cd1(yes)->op1->op4
cd1(no)->op2->op3->op4->op5->op6->ed
```

以上就是本章写的所有内容，下面放出代码来详细解释一下。

## 代码解析

### 主文件

```python
import os
import numpy as np
import OperatePicture as OP
import OperateDataBase as OD
import csv

##Essential vavriable 基础变量
#Standard size 标准大小
N = 100
#Gray threshold 灰度阈值
color = 100/255

#读取原CSV文件
reader = list(csv.reader(open('DataBase.csv', encoding = 'utf-8')))
#清除读取后的第一个空行
del reader[0]
#读取num目录下的所有文件名
fileNames = os.listdir(r"./num/")
#对比fileNames与reader，得到新增的图片newFileNames
newFileNames = OD.NewFiles(fileNames, reader)
print('New pictures are: 'newFileNames)
#得到newFilesNames对应的矩阵
pic = OP.GetTrainPicture(newFileNames)
#将新增图片矩阵存入CSV中
OD.SaveToCSV(pic, newFileNames)
#将原数据库矩阵与新数据库矩阵合并
pic = OD.Combination(reader, pic)
```

我将两节内容分别封装在两个py文件里面，上一篇文章中的图片的切割与处理等所有内容我放在文件OperatePicture里面了，这一节的数据库处理放在了文件OperateDatabase里面。

因为整个代码的逻辑我在上面已经捋过一遍了，所以我不再解释其中的内容，接下来针对每个函数开始讲解。

### OperateDatabase代码

从上面的主文件中，我们首先用到了函数NewFiles，主要是对比fileNames和reader这两个文件中图片的名称有什么不同，返回值是新增的图片的名称的列表。下面是代码

```python
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
```

首先判断reader是否有内容，如果没有内容，说明是第一次执行，那么会直接把fileNames返回。否则才会进入下面进行比较。

返回了newFileNames之后，就会把这个列表中的所有名称的图片通过GetTrainPicture函数得到一个1x10001大小的矩阵，具体过程请看我上一篇文章讲的内容。

之后为了把新的数据存入CSV文件中，我们利用函数SaveToCSV将pic存入文件中，具体代码如下。

```python
def SaveToCSV(pic, fileNames):
    '''将pic与对应的dileNames存入CSV文件'''
    writer = csv.writer(open('Database.csv', 'a', newline = ''), dialect = 'excel')
    #将fileNames变为列表
    f = [item for item in fileNames]
    #每一行依次写入文件中
    for i in range(len(pic)):
        #将改行图片向量转为list
        item = pic[i].tolist()
        #将这个图片向量对应的名称f放入列表最后一个
        item.append(f[i])
        writer.writerow(item)
```

当函数运行过后，会把pic矩阵对应的内容直接给续写入CSV文件中，相当于数据库操纵的写入，并不会覆盖之前原有的数据。

之后我们需要将数据库原有的一大堆数据reader和新加进来的数据pic合并到pic里面，所以利用Combination函数将两个矩阵合并，代码如下

```python
def Combination(reader, pic):
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
```

因为reader最后一行还包括了一个图片的名称，所以先利用pop将其去掉，之后转化为矩阵形式，然后再直接放入矩阵中。这个矩阵操作可能没有见过，下面我详细解释一下。

假如我现在有一个2x3的矩阵和一个2x2的矩阵

```python
m = [[1 2 3]
     [4 5 6]]
n = [[7 8]
     [9 1]]
```

我可以进行如下操作

```python
#操作一
m[:, 0:2] = n
print(m)
#操作二
m[:, 1:3] = n
print(m)

#以下为输出结果
#操作一
[[7 8 3]
 [9 1 6]]
#操作二
[[7 7 8]
 [9 9 1]]
```

可以看出操作一直接把m的第一二列给替换成n，操作二把m的第二三列替换成了n。具体过程可以百度查一下numpy的矩阵的操作，也可以自己总结规律，不细讲了。

以上就是这一篇的全部代码。

## 小结

这一篇我相当于用CSV文件制作了一个非常简陋的数据库，能够执行的操作只有识别已有内容NewFiles与添加内容SaveToCSV，并没有插入、删改等操作。主要是我觉得这两个函数目前已经够用，因此只写了这两个操作，所以再需求已经被满足的情况下就不再拓展了。

所有的源代码已经上传到了[我的GitHub](https://github.com/HanpuLiang/Simple-Handwritten-Numerel-Recogntion)上，可以前去下载，谢谢阅读。

如果喜欢的话麻烦点一个喜欢哦，加关注可以得到超厉害的更新提醒。