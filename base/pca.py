#_*_coding:utf-8_*_

'''
PCA：降维
'''
'''
将数据转换为只保留前N个主成分特征空间
1、去除平均值
2、计算协方差矩阵
3、计算协方差矩阵的特征值和特征向量
4、将特征值排序保留前N个最大特征值对应的特征向量
5、将数据转换到上面得到的N个特征向量构建的特征空间(实现了特征压缩)
'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits


datasets = load_digits().data

def percent2n(eigVals,percent):
    sortArray=np.sort(eigVals)   #升序
    sortArray=sortArray[-1::-1]  #逆转，即降序
    arraySum=sum(sortArray)
    tmp=0
    num=0
    for i in sortArray:
        tmp+=i
        num+=1
        if tmp>=arraySum*percent:
            return num

def pca(datasets,percent=0.99):
    #求每一列的均值   若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    meanVals = np.mean(datasets,axis=0)
    #数据矩阵的每一列减去该列的均值
    meanRemoved = datasets - meanVals
    #计算协方差矩阵，除以n-1是为了得到协方差的无偏估计
    covMat = np.cov(meanRemoved,rowvar=0)
    #计算协方差矩阵的特征值和特征向量       一列一个特征向量
    eigVals,eigVects= np.linalg.eig(np.mat(covMat))#np.mat() 构建矩阵

    n = percent2n(eigVals, percent)
    print(n)
    #argsort():对特征值矩阵进行由大到小的排序，返回对应排序后的索引
    eigValId= np.argsort(eigVals)
    #从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引值
    eigValId = eigValId[:-(n+1):-1]  #这里三个数值分别表示：起始位置：重点位置：步长 最后-1表示倒序
    #将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵     eigValId 抽取数组中对应值的特定列
    redeigVects= eigVects[:,eigValId]
    #将去均值后的数据矩阵*压缩矩阵，转换到新的矩阵空间，使维度降维N
    lowDatasets = meanRemoved * redeigVects
    #利用降维后的矩阵反构出原数矩阵
    reconMat = (lowDatasets * redeigVects.T) + meanVals
    return lowDatasets,reconMat

low,rec = pca(datasets)
print(low.shape)


