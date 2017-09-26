# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:18:10 2017

@author: zzd
"""

import numpy as np
from sklearn.datasets.base import Bunch
import pickle


#自带学习测试集
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him','my'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

#读入Bunch对象
def readBunchObj(path):
    fileObj=open(path,'rb')
    bunch=pickle.load(fileObj)
    fileObj.close()
    return bunch
#朴素贝叶斯分类器
class NBayes(object):
    
    def __init__(self):
        self.vocabulary=[]      #词袋
        self.idf=0              #词的idf权值向量
        self.tf=0               #训练集的词频向量,在用TF-IDF权重策略改进的算法中tf又表示tfidf权重

        self.tdm=0              #p(x|yi)=p(a1|yi)p(a2|yi)……p(am|yi)
        self.py={}              #p(yi)
        self.labels=[]          #对应每个文本的分类，是一个外部导入的列表
        self.docLen=0           #训练集文本数
        self.vocabLen=0         #词袋总长度
        self.testSet=0          #测试集
        
    def fit_transform(self,trainSet,classVec):
        #计算数据集中各个分类的概率p(yi)
        self.calPy(classVec)
        #计算训练集文本数
        self.docLen=len(trainSet)
        #生成词袋
        self.createWordBag(trainSet)
        #计算词袋总长度
        self.vocabLen=len(self.vocabulary)
        #计算数据集词频
        self.calTfidf(trainSet)#self.calTf(trainSet)
        #计算p(x|yi)
        self.calTdm()
        
    #计算在数据集中每个分类的概率p(yi)
    def calPy(self,classVec):
        self.labels=classVec;
        #获取全部分类
        labelSets=set(classVec)
        for label in labelSets:
            self.py[label]=float(self.labels.count(label))/float(len(self.labels))
        #print(self.py)
     
    #生成词袋
    def createWordBag(self,trainSet):
        wordSet=set()
        [wordSet.add(word) for doc in trainSet for word in doc]
        self.vocabulary=list(wordSet)
        #print(self.vocabulary)
    
    #计算词频向量
    def calTf(self,trainSet):
        #词频向量初始化
        self.tf=np.zeros([self.docLen,self.vocabLen])
        for i in range(self.docLen):
            for word in trainSet[i]:
                self.tf[i,self.vocabulary.index(word)]+=1
        #return self.tf
        
    #通过TF-IDF权重策略来改进权重
    def calTfidf(self,trainSet):
        self.tf=np.zeros([self.docLen,self.vocabLen])
        self.idf=np.zeros([1,self.vocabLen])
        for i in range(self.docLen):
            for word in trainSet[i]:
                self.tf[i,self.vocabulary.index(word)]+=1
            #消除不同文件长度带来的偏差
            self.tf[i]=self.tf[i]/float(len(trainSet[i]))
            for signleWord in set(trainSet[i]):
                self.idf[0,self.vocabulary.index(signleWord)]+=1
        self.idf=np.log(float(self.docLen)/self.idf)#自然对数
        self.tf=np.multiply(self.tf,self.idf)#对应元素相乘
           
    #计算p(x|yi)
    def calTdm(self):
        #每个类别下各词的数目
        self.tdm=np.zeros([len(self.py),self.vocabLen])
        #统计每个分类下的总值
        sumList=np.zeros([len(self.py),1])
        
        for i in range(self.docLen):
            #叠加该类别下的各个词的数目
            self.tdm[self.labels[i]]+=self.tf[i]
            #更新该类别下的总值
            sumList[self.labels[i]]=sum(self.tdm[self.labels[i]])
            
        self.tdm=self.tdm/sumList
        
    #将测试数据转化为词袋标准模式   
    def map2vocab(self,testSet):
        self.testSet=np.zeros([1,self.vocabLen])
        for word in testSet:
            self.testSet[0,self.vocabulary.index(word)]+=1
    #预测
    def predict(self,testSet):
        self.map2vocab(testSet)
        if np.shape(self.testSet)[1]!=self.vocabLen:
            print('输入的数据有误！')
        else:
            predValue=0
            predClass=""
            
            for tdm,keyClass in zip(self.tdm,self.py):
                #p(x|yi)*p(yi)
                temp=np.sum(self.testSet*tdm*self.py[keyClass])
                if temp>predValue:
                    predValue=temp
                    predClass=keyClass
        return predClass
         
if __name__=="__main__":
    
    #学习测试案例一
    [postingList,classVec]=loadDataSet()
    b=NBayes()
    b.fit_transform(postingList,classVec)    
    predicted=b.predict(postingList[3])

    
    
    
    
    