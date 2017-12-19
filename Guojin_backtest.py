# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:01:38 2017

@author: DELL
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

test_list=[] #存放测试集的文加路径，测试集的选取见"代码说明.doc"
feavec=[] #存放特征向量名称
for i in range(96):
    feavec.append("feavec"+str(i))#生成特征向量名称


data_list = ["C:/Users/DELL/Desktop/国金基金笔试/rb1801_1minute.csv"] #存放数据集的文件路径
    
train_data=pd.DataFrame()#生成存放训练集的DataFrame
test_data=pd.DataFrame()#对测试集重复上述做法

########################### 构造训练集的特征向量和类别 ##########################

data=(pd.read_csv(data_list[0])).loc[15085:27953]
data = data.reset_index(drop = True)
#下面是基于现有数据生成相应特征值
data['open_diff']= data.open.diff()/data.open  #open_diff为开盘价变化率
data['close_diff']= data.close.diff()/data.close  #close_diff为收盘价变化率
data['high_diff']= data.high.diff()/data.high #high_diff为最高价变化率
data['low_diff']= data.low.diff()/data.low #low_diff为最低价变化率
data['volume_diff']= data.volume.diff()/data.volume #volume_diff为成交量变化率
data['openInterest_diff']= data.openInterest.diff()/data.openInterest #openInterest_diff为未平仓量变化率
data['c_diff']= data.close.diff() #涨跌幅

quantile_1 = data.close_diff.quantile(0.70) #以前30%的变动幅度作为判定上涨的标准
quantile_2 = data.close_diff.quantile(0.30) #以后30%的变动幅度作为判定下跌的标准

def make_label(p):
    if(p>quantile_1):
        return 1
    elif(p<quantile_2):        
        return -1
    else:
        return 0
    #给数据做标签的函数，上涨赋值为1，下跌赋值为-1，其余赋值为0

data['label']=data.close_diff.apply((lambda x:make_label(x))) #label函数为根据涨、跌、平三种情况给数据分别记上1，-1,0的标记
data['price_change_next_open']=-data.open.diff()
    
for t in range(len(data)-18):
    a=[] #生成用于存放特征向量的列表
    a.extend(data['open_diff'][(t+1):(t+17)]) #取该数据前16个数据的p/ma5，加到a中，下面的做法类似
    a.extend(data['close_diff'][(t+1):(t+17)]) 
    a.extend(data['high_diff'][(t+1):(t+17)])
    a.extend(data['low_diff'][(t+1):(t+17)])
    a.extend(data['volume_diff'][(t+1):(t+17)])
    a.extend(data['openInterest_diff'][(t+1):(t+17)])
    adf=pd.DataFrame(data=a).T #将a变为DataFrame并转置
    adf.columns=feavec #给特征向量命名
    adf['date']=data.loc[t+17,'date'] #附上时间
    adf['label']=data.loc[t+17,'label'] #附上标签
    adf['p_change']=data.loc[t+17,'c_diff'] #附上涨跌额
    adf['price_change']=data.loc[t+17,'close_diff'] #附上涨跌幅
    adf['Num']=t+17 #附上时间编号
    adf['open']=data.loc[t+17,'open'] #附上开盘价  
    adf.dropna(inplace=True)#如果特征向量有缺失数据，就扔掉
    adf['price_change_next_open']=data.loc[t+17,'price_change_next_open']#附上下一条数据的开盘价
    train_data=pd.concat([train_data,adf])

########################### 构造测试集的特征向量和类别 ##########################

testdata=(pd.read_csv(data_list[0])).loc[27954:29333]
testdata = testdata.reset_index(drop = True)
#下面是基于现有数据生成相应特征值
testdata['open_diff']= testdata.open.diff()/testdata.open  #open_diff为开盘价变化率
testdata['close_diff']= testdata.close.diff()/testdata.close  #close_diff为收盘价变化率
testdata['high_diff']= testdata.high.diff()/testdata.high #high_diff为最高价变化率
testdata['low_diff']= testdata.low.diff()/testdata.low #low_diff为最低价变化率
testdata['volume_diff']= testdata.volume.diff()/testdata.volume #volume_diff为成交量变化率
testdata['openInterest_diff']= testdata.openInterest.diff()/testdata.openInterest #openInterest_diff为未平仓量变化率
testdata['c_diff']= testdata.close.diff() #涨跌幅

testdata['price_change_next_open']=-testdata.open.diff()
    
for t in range(len(testdata)-18):
    a_test=[] #生成用于存放特征向量的列表
    a_test.extend(testdata['open_diff'][(t+1):(t+17)]) #取该数据前16个数据的p/ma5，加到a中，下面的做法类似
    a_test.extend(testdata['close_diff'][(t+1):(t+17)]) 
    a_test.extend(testdata['high_diff'][(t+1):(t+17)])
    a_test.extend(testdata['low_diff'][(t+1):(t+17)])
    a_test.extend(testdata['volume_diff'][(t+1):(t+17)])
    a_test.extend(testdata['openInterest_diff'][(t+1):(t+17)])
    adf_test=pd.DataFrame(data=a_test).T #将a变为DataFrame并转置
    adf_test.columns=feavec #给特征向量命名
    adf_test['date']=testdata.loc[t+17,'date'] #附上时间
    adf_test['p_change']=testdata.loc[t+17,'c_diff'] #附上涨跌额
    adf_test['price_change']=testdata.loc[t+17,'close_diff'] #附上涨跌幅
    adf_test['Num']=t+17 #附上时间编号
    adf_test['open']=testdata.loc[t+17,'open'] #附上开盘价  
    # adf_test.dropna(inplace=True)#如果特征向量有缺失数据，就扔掉
    adf_test['price_change_next_open']=testdata.loc[t+17,'price_change_next_open']#附上下一条数据的开盘价
    test_data=pd.concat([test_data,adf_test])

###################### 训练模型 ###############################################
train_feavec=train_data[feavec].values #将train_data的特征取出
train_label=train_data['label'].values #将train_data的标记取出
test_feavec=test_data[feavec].values #将test_data的特征取出
# test_label=test_data['label'].values #将test_data的标记取出 

rf1=RandomForestClassifier(n_estimators=100) #构建一个由30棵决策树组成的随机森林
rf1.fit(train_feavec,train_label) #训练模型
predict_proba=rf1.predict_proba(test_feavec) #对测试集做出预测
pro_df=pd.DataFrame(predict_proba) #将预测结果转为dataFrame

###################### 回测 ###############################################
position = 0 #仓位
initialMoney = 1000000

curMoney = initialMoney #当期总资产
lastMoney = initialMoney #上一期总资产, 计算每期盈亏

positive_profit = []
negative_profit = []
long_profit = []
short_profit = []
Indi = 0.2

for i in range(0,len(pro_df)) :
    long_predict = pro_df.values[i,0]
    short_predict = pro_df.values[i,2]
    Ride_Mood = long_predict - short_predict
    if ( i == len(pro_df)-1 ) and ( position != 0 ): #期末强制平仓
        curMoney += abs(position)*testdata.close[i+18]
        profit = curMoney - lastMoney
        position = 0
        print("Time is out, shut down at %s, price is %f, profit is %f" %(testdata.date[i+18],testdata.close[i+18],profit) )
    elif (position == 0) and (Ride_Mood > Indi) :
        lastMoney = curMoney
        position =  int( curMoney/testdata.close[i+18] )  
        curMoney = lastMoney - position*testdata.close[i+18]
        print("Long at %f at %s %s" %(testdata.close[i+18], testdata.date[i+18],testdata.time[i+18]) )
    elif (position == 0) and (Ride_Mood < -Indi) :
        lastMoney = curMoney
        position =  -int( curMoney/testdata.close[i+18] )  
        curMoney = lastMoney + position*testdata.close[i+18]
        print("short at %f at %s %s" %(testdata.close[i+18], testdata.date[i+18],testdata.time[i+18]) )
    elif (position > 0) and (Ride_Mood < -Indi) and (long_predict<0.3) :
        curMoney += position*testdata.close[i+18]
        profit = curMoney - lastMoney
        lastMoney = curMoney
        position = -int( curMoney/testdata.close[i+18] ) 
        curMoney = lastMoney + position*testdata.close[i+18]
        print("short at %f at %s %s, profit is %f" %(testdata.close[i+18],testdata.date[i+18],testdata.time[i+18],profit))
        short_profit.append(profit)
        if(profit > 0):
            positive_profit.append(profit)
        else:
            negative_profit.append(profit)
    elif (position < 0) and (Ride_Mood > Indi) and (short_predict<0.3):
        curMoney -= position*testdata.close[i+18]
        profit = curMoney - lastMoney
        lastMoney = curMoney
        position = int( curMoney/testdata.close[i+18] ) 
        curMoney = lastMoney - position*testdata.close[i+18]
        print("long at %f at %s %s, profit is %f" %(testdata.close[i+18],testdata.date[i+18],testdata.time[i+18],profit))
        long_profit.append(profit)
        if(profit > 0):
            positive_profit.append(profit)
        else:
            negative_profit.append(profit)


print ("收益率为 %f" %(curMoney/initialMoney-1) )
print ("盈利数目 %d, 平均每笔盈利 %f"%(len(positive_profit),sum(positive_profit)/(initialMoney*len(positive_profit)) ))
print ("亏损数目 %d, 平均每笔亏损 %f"%(len(negative_profit),sum(negative_profit)/(initialMoney*len(negative_profit)) ))
print ("多仓平均每笔盈利 %f"%(sum(long_profit)/(initialMoney*len(long_profit)) ))
print ("空仓平均每笔盈利 %f"%(sum(short_profit)/(initialMoney*len(short_profit)) ))