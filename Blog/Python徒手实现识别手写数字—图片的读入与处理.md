# 写在前面

在上一篇文章[Python徒手实现手写数字识别—大纲](http://www.jianshu.com/p/09e735d6a2df)中，我们已经讲过了我们想要写的全部思路，所以我们不再说全部的思路。

我这一次将图片的读入与处理的代码写了一下，和大纲写的过程一样，这一段代码分为以下几个部分：

- 读入图片；
- 将图片读取为灰度值矩阵；
- 图片背景去噪；
- 切割图片，得到手写数字的最小矩阵；
- 拉伸/压缩图片，得到标准大小为100x100大小矩阵；
- 将图片拉为1x10000大小向量，存入训练矩阵中。

所以下面将会对这几个函数进行详解。

# 代码分析

## 基础内容

首先我们现在最前面定义基础变量

```python
import os
from skimage import io
import numpy as np

##Essential vavriable 基础变量
#Standard size 标准大小
N = 100
#Gray threshold 灰度阈值
color = 100/255
```

其中标准大小指的是我们在最后经过切割、拉伸后得到的图片的尺寸为NxN。灰度阈值指的是在某个点上的灰度超过阈值后则变为1.

接下来是这图像处理的一部分的主函数

```python
filenames = os.listdir(r"./num/")
pic = GetTrainPicture(filenames)
```

其中filenames得到在num目录下所有文件的名称组成的列表。pic则是通过函数GetTrainPicture得到所有训练图像向量的矩阵。这一篇文章主要就是围绕这个函数进行讲解。

## GetTrainPicture函数

GetTrainPicture函数内容如下

```python
def GetTrainPicture(files):
    '''Read and save train picture 读取训练图片并保存'''
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
```

可以看出这个函数的信息量非常大，基本上今天做的所有步骤我都把封装到一个个函数里面了，所以这里我们可以看到图片处理的所有步骤都在这里。

### 提前准备

首先是创建了一个用来存放所有图像向量的矩阵Picture，大小为fx10001，其中f代表我们拥有的训练图片的数目，10001的前10000位代表图片展开后的向量长度，最后一维代表这一个向量的类别，比如说时2就代表这个图片上面写的数字是2.

接下来用的是一个for循环，将files里面每一个图片进行一次迭代，计算出向量后存入picture。

在循环中的内容就是对每一张图片进行的操作。

### 读入图片并清除背景噪音

首先是io.imread函数，这个函数是将图片导出成为灰度值的矩阵，每一个像素点是矩阵上的一个元素。

接下来是img[img>color]=1这一句。这一句运用了逻辑运算的技巧，我们可以将其分为两部分

```python
point = img > color
img[point] = 1
```

首先是img>color，img是一个矩阵，color是一个数。意义就是对img中所有元素进行判断是否大于color这个数，并输出一个与img同等大小的矩阵，对应元素上是**该值与color判断后的结果**，有False与True。如果大于这个数，那么就是Ture，否则是False。下面举个例子，不再赘述。

```python
a = np.array([1, 2, 3, 4])
print(a>2)

#以下为输出结果
[False False True True]
```

之后的img[point] = 1说明**将所有True的值等于1**。举个例子

```python
a = np.array([1, 2, 3, 4])
p = a > 2
a[p] = 0
print(a)

#以下为输出结果
[1 2 0 0]
```

因此我通过这样的方法来清除掉了与数字颜色差别太大的背景噪音。

### 切割图像

首先切割图像的函数我写的是CutPicture。我们来说一下这个切割图像的意思。比如说有一个人写字写的特别小，另一个人写字写的特别大。就像是下图所示，所以我们进行这样的操作。沿着图片的边进行切割，得到了下面切割后的图片，让数字占满整个图片，从而具有可比性。

![](http://a4.qpic.cn/psb?/V149PbC91ayDNp/Gj*5K*TbZG.42r7WxEqHSJAEuaB8LDg58SKgODdAzYc!/b/dFsBAAAAAAAA&ek=1&kp=1&pt=0&bo=egJhAgAAAAADFyk!&vuin=798258079&tm=1507906800&sce=60-2-2&rf=viewer_4)

所以下面贴出代码，详细解释一下我是怎么做的。

```python
def CutPicture(img):
    '''Cut the Picture 切割图象'''
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
```

首先函数导入过来的的参数只有一个原img。

我第一步做的是把新的大小初始化一下，size一共会放入四个值，第一个值代表原图片上的**手写数字图案的最高行**，第二个值代表的是**最低行**，第三个值代表**数字图案的最左列，，第四个只代表**最右列**。这个还看不明白的话就看上面的图示，就是沿着图片切割一下就好了。

接下来的length和width分别代表着原图片的行数与列数，作用在下面。我又创建了一个JudgeEdge函数，这个函数是输出它的行数或者列数的两位数字。第一个append是给size列表放入了两个行序号（最高行和最低行），第二个append是给size放进两个列序号（最左列和最右列）。所以接下来就看JudgeEdge函数是干什么的。

```python
def JudgeEdge(img, length, flag, size):
    '''Judge the Edge of Picture判断图片切割的边界'''
    for i in range(length):
        #Row or Column 判断是行是列
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
```

JudgeEdge函数的参数flag就是用来判断是行还是列，当flag=0时，说明以行为基础开始循环；当flag=1时说明以列为基础进行循环。所以参数length传递的时候就是行数和列数。接下来的for循环就是根据length的大小看似是循环。

当是行时，我这里用line1和line2起到的是一个指针的作用，即在第i行时，line1的内容就是这一行拥有非白色底（数值为1）的像素的个数；line2的作用则是**反序的**，也就是他计算的是**倒数i行**非白色像素个数，这样做的目的是能够快一点，让上下同时开始进行寻找，而不用line1把整个图片循环一遍，line2把整个图片循环一遍，大大节省了时间。

寻找这一行拥有非白色底的像素的个数这一个语句同样的运用了逻辑判断，和上文中去噪的原理一模一样。

当line里面有数的时候，说明已经到达了手写数字的边缘。这时候就记录下来，然后就不再改变。当两个line都不等于初始值-1时，说明已经找到了两个边缘，这时候就可以跳出循环并且return了。

这个函数就是这样。所以说切割图像的完整代码就是这样子，下面就要把切割的大小不一的图像给拉伸成标准大小100x100了。

### 拉伸图像

因为切割以后的图像有大有小，一张图片的大小可能是21x39（比如说数字2），另一张可能是4x40（比如说数字1）。所以为了能够让他们统一大小称为100x100，我们就要把他们拉伸一下。

![](http://a2.qpic.cn/psb?/V149PbC91ayDNp/m4miZmpqlhHLLgBxm689cKYagfV5IDJOCYRHw..PUTg!/b/dGkBAAAAAAAA&bo=vQKAAgAAAAADBx8!&rf=viewer_4)

大概就是像图上的一样。实际情况的图案可能会更复杂，所以我们下面展示一下代码

```python
def StretchPicture(img):
    '''Stretch the Picture拉伸图像'''
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
```

首先初始化一个新的图片矩阵，这个大小就是标准大小100x100。接下来才是重头戏。我这里用的方法是比较简单基础的方法，但是可能依旧比较难。

首先定义两个步长step1和step2，分别代表拉伸/压缩行与列时的步长。这里的原理就是把原来的长度给他平均分成100份，然后将这100个像素点分别对应上原本的像素点。

如下图所示，图像1我们假设为原图像，图像2我们假设为标准图像，我们需要把图像1转化为图像2，其中每一个点代表一个像素点，也就是图像1有五个像素点，图像2有四个像素点。

![](http://a1.qpic.cn/psb?/V149PbC91ayDNp/xuEEWSA7I9Pv0KgiT3JTREc1RTFrkWnSlomcKvqFqk8!/b/dPMAAAAAAAAA&bo=bQPOAAAAAAADB4I!&rf=viewer_4)

所以我的思想就是直接让图像2的像素点的值等于距离它最近的图像1的像素点。

我们为了方便起见，在这里定义一个语法：图像2第三个数据点我们可以写为2_3.

所以2_1对应的就是1_1，2_2对应的就是1_2，2_3对应的是1_4，2_4对应的是1_5。就这样我们就能够得到了图像2所有的数据点。

利用数学的形式表现出来，就是假设图像1长度为l_1，图像2长度为l_2，所以令图像2的步长为l_1/l_2，也就是说当图像2的第一个像素点对应图像1第一个像素点，图像2的最后一个像素点对应图像1最后一个像素点。然后图像2第二个像素点位置就是2*l_1/l_2，对应图像1第floor(2*l_1/l_2)个像素点。以此类推就行。因此再回头看一下那一段代码，这一段是不是就好理解了？

之后对行与列分别进行这个操作，所以就可以得到标准的图片大小。然后再返回到GetTrainPicture即可。

再GetTrainPicture函数中，我用了reshape函数把原本100x100大小的图片拉伸成为1x10000大小的向量，然后存入矩阵当中，并将这一张图片的类别存入矩阵最后一个。

以上就是图片处理的所有内容。

# 小结

以上就是把一张图片经过处理后存入矩阵的内容。

本文中的所有算法、代码均是我自己构思的，所以可能存在一些不足之处，我没有系统的学习过图像的相关知识，也并不是计算机专业，因此可能在理论上有一些不合乎情况，所以如果有错误的话欢迎一起讨论，谢谢。

目前所有源代码都在[我的GitHub](https://github.com/HanpuLiang/Simple-Handwritten-Numerel-Recogntion)中更新哦。

如果喜欢的话，麻烦点一个喜欢哦，谢谢！