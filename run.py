import numpy as np
from PIL import Image
import os

#初始化要读进图片的数据空间 15组数据，每组11张图片，图片大小320*243
f=np.zeros((15,12,77760),dtype='float64')
for i in range(15):
    path='yalefaces\\' + str(i+1)
    j=-1
    for name_file in os.listdir(path):
        j+=1
        #读图片
        m = Image.open(path +'\\' +name_file)
        #转成数组
        img2 = np.array(m)
        #把二维变成一维
        img1=img2.reshape(-1)
        #拷贝到要存的结构中
        for ii in range(0,77760):
            f[i][j][ii]=img1[ii]

#中心化
for i in range(15):
    for j in range(77760):
        for k in range(10):
            f[i][11][j]+=f[i][k][j]
        f[i][11][j]/=10
        for k in range(10):
            f[i][k][j]-=f[i][11][j]
    #print(f[i])
    a=f[i,0:10] 
    #print(a)
    #print(a.shape)
    cov=np.dot(a,a.T)
    #print(cov)
    #print(cov.shape)
    eigvalue, eigvector = np.linalg.eig(cov)
    #print(eigvalue)
    #print(eigvector)
    #print(eigvector.shape)
    cova=np.dot(eigvector,a)
    for j in range(10):
        for k in range(77760):
            f[i][j][k]=cova[j][k]

#测试
r=0.0
for t in range(15):
    ar=np.zeros((15,77760),dtype='float64')
    min=0
    mi=0
    a1=f[t,10]
    #print(a.shape)
    for i in range(15):
        #print(a.shape)
        b1=a1-f[i][11]
        for j in range(10):
            #print(a1.shape)
            #print(a1.T.shape)
            #print(f[i][j].shape)
            #print(f[i][j].T.shape)
            #print("%%%%%")
            b=b1.dot(f[i][j])/np.sum(np.square(f[i][j]))
            #print("b")
            #print(b)
            ar[i]+=b*f[i][j]
            #print(ar[i])
            #print(ar[i].shape)
        m=np.sum(np.square(ar[i]-b1))
        #print(ar[i])
        #print(ar[i].shape)
        #print(a)
        #print(a.shape)
        if(min == 0):
            min=m
            mi=i+1
        elif(m<min):
            min=m
            mi=i+1
    if (t+1==mi):
            r+=1
    print("样本"+str(t+1)+"识别为"+str(mi))
r/=15.0
print("正确率"+str(round(r,2)))