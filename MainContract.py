# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 08:39:40 2017

@author: DELL
"""

import pandas as pd
import numpy as np
import scipy.io as sio

######################### 提取主力合约列表 #########################

RB = u'C:/Users/DELL/Desktop/国金基金笔试/RB.mat' #主力合约地址
RBdata_temp = sio.loadmat(RB)
#str(RBdata) #查看可提取的字符串
RBdata_temp = RBdata_temp['MainContract']

RBdata = pd.DataFrame() #存放主力合约列表
Column_Name = ['name','date','time','open','high','low','close','volume','turnover','NAN','average','openInterest']
Alldata = pd.DataFrame() #存放主力合约详细信息的汇总

RBname = [] #存放所有主力合约的代码
RBdate = [] #存放相应的日期
for i in range(len(RBdata_temp)):
    RBname.append( RBdata_temp[i][0].tolist()[0] )
    RBdate.append( RBdata_temp[i][1].tolist()[0] )
    
RBdata['name'] = RBname
RBdata['date'] = RBdate
#set(RBname) #需要读取的期货合约
RBlist = list(set(RBname))
RBlist.sort(key = RBname.index)

######################### 合成主力合约数据 #########################

for i in RBlist:
    temp_site = ["C:/Users/DELL/Desktop/国金基金笔试/15min/RB/"+i+".mat"]
    data_temp = sio.loadmat(temp_site[0])
    data_temp = data_temp['MinData']
    start_date = RBdata[RBdata['name']==i].date.min() #该主力合约开始日期
    end_date = RBdata[RBdata['name']==i].date.max() #该主力合约结束日期
    for j in range(len(data_temp)):
        #在end_date之后的直接结束循环
        IsBreak = False    
        if ( str(int(data_temp[j][0])) >= start_date ) and ( str(int(data_temp[j][0])) <= end_date ):
            temp = [i] #保存主力合约对应的名称
            for k in range(len(data_temp[j])):
                if k==0 or k==1:
                    temp.append(str(int(data_temp[j][k])))  #date和time用字符串表示
                else:
                    temp.append(data_temp[j][k])
            temp_frame = pd.DataFrame(data = temp).T
            temp_frame.columns = Column_Name
            Alldata = pd.concat([Alldata, temp_frame])
            IsBreak = True
        elif IsBreak == True:
            break

Alldata = Alldata.reset_index(drop = True) #赋予正确的index

Alldata.to_csv("C:/Users/DELL/Desktop/国金基金笔试/MainContract.csv", index=False)

