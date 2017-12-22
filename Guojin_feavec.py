# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 19:02:15 2017

@author: DELL
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import sklearn
import talib as ta

data = pd.read_csv("C:/Users/DELL/Desktop/国金基金笔试/MainContract.csv")
data['close_diff'] = data.close.diff()/data.close #涨跌幅

##################### 计算指标 ############################

shortperiod = 5
medperiod = 10
longperiod = 20

close = np.array([float(x) for x in data.close.values])
high = np.array([float(x) for x in data.high.values])
low = np.array([float(x) for x in data.low.values])
volume = np.array([float(x) for x in data.volume.values])

price = np.array([high,low,close]) #收集所有价格数据
price_list = ['high','low','close']
feavec=[] #存放特征向量名称

for i in range(len(price)):
    data['EMA_'+price_list[i]] = ta.EMA(price[i], timeperiod = medperiod)
    data['DEMA_'+price_list[i]] = ta.DEMA(price[i], timeperiod = medperiod)
    data['DIF_'+price_list[i]] = ta.MACD(price[i], fastperiod = shortperiod, slowperiod = longperiod)[0]
    data['DEA_'+price_list[i]] = ta.MACD(price[i], fastperiod = shortperiod, slowperiod = longperiod)[1]
    data['MACD_'+price_list[i]] = ta.MACD(price[i], fastperiod = shortperiod, slowperiod = longperiod)[2]
    data['TRIX_'+price_list[i]] = ta.TRIX(price[i], timeperiod = longperiod)
    data['RSI_'+price_list[i]] = ta.RSI(price[i], timeperiod = longperiod)
    data['ROC_'+price_list[i]] = ta.ROC(price[i], timeperiod = medperiod)
    data['STD_'+price_list[i]] = ta.STDDEV(price[i], timeperiod = shortperiod)
    data['OBV_'+price_list[i]] = ta.OBV(price[i], volume)
    for j in range(-3,0):
        feavec.append('EMA_'+price_list[i]+str(j))
        feavec.append('DEMA_'+price_list[i]+str(j))
        feavec.append('DIF_'+price_list[i]+str(j))
        feavec.append('DEA_'+price_list[i]+str(j))
        feavec.append('MACD_'+price_list[i]+str(j))
        feavec.append('TRIX_'+price_list[i]+str(j))
        feavec.append('RSI_'+price_list[i]+str(j))
        feavec.append('ROC_'+price_list[i]+str(j))
        feavec.append('STD_'+price_list[i]+str(j))
        feavec.append('OBV_'+price_list[i]+str(j))
        
# 其它指标
data['SAR'] = ta.SAR(high, low)
data['CCI'] = ta.CCI(high, low, close, timeperiod = medperiod)
data['AD'] = ta.AD(high, low, close, volume)
data['ADOSC'] = ta.ADOSC(high, low, close, volume, fastperiod = shortperiod, slowperiod = longperiod)
for j in range(-3,0):
    feavec.append('SAR_'+str(j))
    feavec.append('CCI_'+str(j))
    feavec.append('AD_'+str(j))
    feavec.append('ADOSC_'+str(j))    

##################### 训练集 & 测试集 ############################
train_start_list = [20100201, 20100501, 20100801, 20101101, 20110201, 20110501, 20110801, 20111101, 20120201, 20120501, 20120801, 20121101, 20130201, 20130501, 20130801] #分期进行回测, 每期长半年, 按5:1划分 
train_end_list = [20100631, 20100931, 20101231, 20110331, 20110631, 20110931, 20111231, 20120331, 20120631, 20120931, 20121231, 20130331, 20130631, 20130931, 20131231] #训练集结束日期, 每期长半年, 按5:1划分 
test_end_list = [20100731, 20101031, 20110131, 20110431, 20110731, 20111031, 20120131, 20120431, 20120731, 20121031, 20130131, 20130431, 20130731, 20131031, 20140131] #测试集结束日期, 每期长半年, 按5:1划分 
sample_accuracy = [] #训练集正确率
predict_accuracy = [] #测试集正确率
importance = pd.DataFrame() #特征重要性矩阵

def make_label(p,q_1,q_2):
    if(p>q_1):
        return 1
    elif(p<q_2):        
        return -1
    else:
        return 0
    #给数据做标签的函数，上涨赋值为1，下跌赋值为-1，其余赋值为0

for i in range(len(train_start_list)):    
    train_start = data[data['date']>=train_start_list[i]].index.min() #训练集从当年2月开始
    train_end = data[data['date']<= (train_end_list[i]) ].index.max() #持续到2年后的9月底
    train_data = data.loc[train_start:train_end]
    train_data = train_data.reset_index(drop = True)
    
    quantile_1 = train_data.close_diff.quantile(0.66) #前30%的变动幅度作为判定上涨的标准
    quantile_2 = train_data.close_diff.quantile(0.33) #后30%的变动幅度作为判定下跌的标准
    train_data['label'] = train_data.close_diff.apply((lambda x:make_label(x, quantile_1, quantile_2)))
    train_feavec = pd.DataFrame(columns=feavec)
    for j in range(-3,0):
        for k in range(len(price)):
            train_feavec['EMA_'+price_list[k]+str(j)] = train_data['EMA_'+price_list[k]] [3+j:j].reset_index(drop = True)
            train_feavec['DEMA_'+price_list[k]+str(j)] = train_data['DEMA_'+price_list[k]] [3+j:j].reset_index(drop = True)
            train_feavec['DIF_'+price_list[k]+str(j)] = train_data['DIF_'+price_list[k]] [3+j:j].reset_index(drop = True)
            train_feavec['DEA_'+price_list[k]+str(j)] = train_data['DEA_'+price_list[k]] [3+j:j].reset_index(drop = True)
            train_feavec['MACD_'+price_list[k]+str(j)] = train_data['MACD_'+price_list[k]] [3+j:j].reset_index(drop = True)
            train_feavec['TRIX_'+price_list[k]+str(j)] = train_data['TRIX_'+price_list[k]] [3+j:j].reset_index(drop = True)
            train_feavec['RSI_'+price_list[k]+str(j)] = train_data['RSI_'+price_list[k]] [3+j:j].reset_index(drop = True)
            train_feavec['ROC_'+price_list[k]+str(j)] = train_data['ROC_'+price_list[k]] [3+j:j].reset_index(drop = True)
            train_feavec['STD_'+price_list[k]+str(j)] = train_data['STD_'+price_list[k]] [3+j:j].reset_index(drop = True)
            train_feavec['OBV_'+price_list[k]+str(j)] = train_data['OBV_'+price_list[k]] [3+j:j].reset_index(drop = True)
        train_feavec['SAR_'+str(j)] = train_data['SAR'][3+j:j].reset_index(drop = True)
        train_feavec['CCI_'+str(j)] = train_data['CCI'][3+j:j].reset_index(drop = True)
        train_feavec['AD_'+str(j)] = train_data['AD'][3+j:j].reset_index(drop = True)
        train_feavec['ADOSC_'+str(j)] = train_data['ADOSC'][3+j:j].reset_index(drop = True)  
    train_label = train_data['label'][3:].reset_index(drop = True)
    
    train_feavec = train_feavec.values
    train_label = train_label.values
 ##################### 测试集 ############################   
    test_start = data[data['date']>=(train_end_list[i]+1)].index.min() #训练集从当年2月开始
    test_end = data[data['date']<= test_end_list[i] ].index.max() #持续到2年后的9月底
    test_data = data.loc[test_start:test_end]
    test_data = test_data.reset_index(drop = True)
    
    quantile_1 = test_data.close_diff.quantile(0.66) #前30%的变动幅度作为判定上涨的标准
    quantile_2 = test_data.close_diff.quantile(0.33) #后30%的变动幅度作为判定下跌的标准
    test_data['label'] = test_data.close_diff.apply((lambda x:make_label(x, quantile_1, quantile_2)))
    test_feavec = pd.DataFrame(columns=feavec)
    for j in range(-3,0):
        for k in range(len(price)):
            test_feavec['EMA_'+price_list[k]+str(j)] = test_data['EMA_'+price_list[k]] [3+j:j].reset_index(drop = True)
            test_feavec['DEMA_'+price_list[k]+str(j)] = test_data['DEMA_'+price_list[k]] [3+j:j].reset_index(drop = True)
            test_feavec['DIF_'+price_list[k]+str(j)] = test_data['DIF_'+price_list[k]] [3+j:j].reset_index(drop = True)
            test_feavec['DEA_'+price_list[k]+str(j)] = test_data['DEA_'+price_list[k]] [3+j:j].reset_index(drop = True)
            test_feavec['MACD_'+price_list[k]+str(j)] = test_data['MACD_'+price_list[k]] [3+j:j].reset_index(drop = True)
            test_feavec['TRIX_'+price_list[k]+str(j)] = test_data['TRIX_'+price_list[k]] [3+j:j].reset_index(drop = True)
            test_feavec['RSI_'+price_list[k]+str(j)] = test_data['RSI_'+price_list[k]] [3+j:j].reset_index(drop = True)
            test_feavec['ROC_'+price_list[k]+str(j)] = test_data['ROC_'+price_list[k]] [3+j:j].reset_index(drop = True)
            test_feavec['STD_'+price_list[k]+str(j)] = test_data['STD_'+price_list[k]] [3+j:j].reset_index(drop = True)
            test_feavec['OBV_'+price_list[k]+str(j)] = test_data['OBV_'+price_list[k]] [3+j:j].reset_index(drop = True)
        test_feavec['SAR_'+str(j)] = test_data['SAR'][3+j:j].reset_index(drop = True)
        test_feavec['CCI_'+str(j)] = test_data['CCI'][3+j:j].reset_index(drop = True)
        test_feavec['AD_'+str(j)] = test_data['AD'][3+j:j].reset_index(drop = True)
        test_feavec['ADOSC_'+str(j)] = test_data['ADOSC'][3+j:j].reset_index(drop = True)  
    test_label = test_data['label'][3:].reset_index(drop = True)
    
    test_feavec = test_feavec.values
    test_label = test_label.values
###################### 训练模型 ############################   
    scaler = sklearn.preprocessing.StandardScaler().fit(train_feavec) #按训练集方式标准化数据
    train_feavec = scaler.transform(train_feavec)
    test_feavec = scaler.transform(test_feavec)    
    
    rf1=RandomForestClassifier(n_estimators=100) #构建一个由100棵决策树组成的随机森林
    rf1.fit(train_feavec,train_label) #训练模型
    
    sample_proba=rf1.predict_proba(train_feavec) #对训练集做出预测
    sample_df=pd.DataFrame(sample_proba, columns=['-1','0','1']) #将预测结果转为dataFrame
    sample_label = sample_df.values.argmax(axis=1)-1 
    sample_accuracy.append( np.sum(sample_label == train_label)/len(train_label) )
    
    predict_proba=rf1.predict_proba(test_feavec) #对测试集做出预测
    pro_df=pd.DataFrame(predict_proba, columns=['-1','0','1']) #将预测结果转为dataFrame
    predict_label = pro_df.values.argmax(axis=1)-1
    predict_accuracy.append( np.sum(predict_label == test_label)/len(test_label) )
    
    importance[str(train_start_list[i])] = rf1.feature_importances_

importance['mean'] = importance.mean(axis=1)
importance = importance.sort_values(by=['mean'],ascending=False)

feavec_sorted = []
for i in range(len(feavec)):
    feavec_sorted.append(feavec[importance.index[i]])

feavec_sorted = pd.DataFrame(data = feavec_sorted, columns = ['feavec'])
feavec_sorted.to_csv("C:/Users/DELL/Desktop/国金基金笔试/feavec_sorted.csv", index=False)
importance.to_csv("C:/Users/DELL/Desktop/国金基金笔试/importance.csv", index=False)


