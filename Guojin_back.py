# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:32:21 2017

@author: DELL
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import sklearn
import talib as ta

data = pd.read_csv("C:/Users/DELL/Desktop/国金基金笔试/MainContract.csv")
feavec_sorted = pd.read_csv("C:/Users/DELL/Desktop/国金基金笔试/feavec_sorted.csv")
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
train_start_list = [20140201, 20140501, 20140801, 20141101, 20150201, 20150501, 20150801, 20151101, 20160201, 20160501, 20160801, 20161101, 20170201, 20170501] #分期进行回测, 每期长半年, 按5:1划分 
train_end_list = [20140631, 20140931, 20141231, 20150331, 20150631, 20150931, 20151231, 20160331, 20160631, 20160931, 20161231, 20170331, 20170631, 20170931] #训练集结束日期, 每期长半年, 按5:1划分 
test_end_list = [20140731, 20141031, 20150131, 20150431, 20150731, 20151031, 20160131, 20160431, 20160731, 20161031, 20170131, 20170431, 20170731, 20171031] #测试集结束日期, 每期长半年, 按5:1划分 
return_rate = []

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
    
    test_label = test_label.values
    
###################### 训练模型 ############################
    test_feavec_N = pd.DataFrame()
    train_feavec_N = pd.DataFrame()
    for i in feavec_sorted.feavec[0:16+1]:
        test_feavec_N[i] = test_feavec[i]
        train_feavec_N[i] = train_feavec[i]
        
    scaler = sklearn.preprocessing.StandardScaler().fit(train_feavec_N) #按训练集方式标准化数据
    train_feavec = scaler.transform(train_feavec_N)
    test_feavec = scaler.transform(test_feavec_N)    

    rf1=RandomForestClassifier(n_estimators = 40) #遍历决策树参数的随机森林
    rf1.fit(train_feavec_N,train_label) #训练模型
    
    sample_proba=rf1.predict_proba(train_feavec_N) #对训练集做出预测
    sample_df=pd.DataFrame(sample_proba, columns=['-1','0','1']) #将预测结果转为dataFrame
    sample_label = sample_df.values.argmax(axis=1)-1 
    
    predict_proba=rf1.predict_proba(test_feavec_N) #对测试集做出预测
    pro_df=pd.DataFrame(predict_proba, columns=['-1','0','1']) #将预测结果转为dataFrame
    predict_label = pro_df.values.argmax(axis=1)-1

###################### 回测 ###############################################
    position = 0 #仓位
    initialMoney = 1000000
    cost = 1.0001 #手续费万1
    
    curMoney = initialMoney #当期总资产
    lastMoney = initialMoney #上一期总资产, 计算每期盈亏
    last_date = []
    
    positive_profit = []
    negative_profit = []
    long_profit = []
    short_profit = []
    Indi = 0.2
    
    for i in range(0,len(pro_df)) :
        long_predict = pro_df.values[i,0]
        short_predict = pro_df.values[i,2]
        Ride_Mood = long_predict - short_predict
        if test_data.date[i] == test_data.date.max():
            if ( i == len(pro_df)-1 ) and ( position != 0 ): #期末强制平仓
                curMoney += position*test_data.close[i+1]
                profit = curMoney - lastMoney
                position = 0
                print("Time is out, shut down at %s, price is %f, profit is %f" %(test_data.date[i+1],test_data.close[i+1],profit) )
            elif position == 0: #最后一日不开仓，避免日内平仓
                break
        elif (position == 0) and (Ride_Mood > Indi) :
            lastMoney = curMoney
            position =  int( curMoney/(test_data.close[i+1]*cost) )  
            curMoney = lastMoney - position*test_data.close[i+1]
            print("Long at %f at %s %s" %(test_data.close[i+1], test_data.date[i+1],test_data.time[i+1]) )
            last_date.append(test_data.date[i+1])
        elif (position == 0) and (Ride_Mood < -Indi):
            lastMoney = curMoney
            position =  -int( curMoney/(test_data.close[i+1]*cost) )  
            curMoney = lastMoney - position*test_data.close[i+1]
            print("short at %f at %s %s" %(test_data.close[i+1], test_data.date[i+1],test_data.time[i+1]) )
            last_date.append(test_data.date[i+1])
        elif (position > 0) and (Ride_Mood < -Indi) and (test_data.date[i+1]>last_date[-1]) and (short_predict>0.4):
            curMoney += position*test_data.close[i+1]
            profit = curMoney - lastMoney
            lastMoney = curMoney
            position = -int( curMoney/(test_data.close[i+1]*cost) ) 
            curMoney = lastMoney - position*test_data.close[i+1]
            print("short at %f at %s %s, profit is %f" %(test_data.close[i+1],test_data.date[i+1],test_data.time[i+1],profit))
            short_profit.append(profit)
            last_date.append(test_data.date[i+1])
            if(profit > 0):
                positive_profit.append(profit)
            else:
                negative_profit.append(profit)
        elif (position < 0) and (Ride_Mood > Indi) and (test_data.date[i+1]>last_date[-1]) and (long_predict>0.4):
            curMoney += position*test_data.close[i+1]
            profit = curMoney - lastMoney
            lastMoney = curMoney
            position = int( curMoney/(test_data.close[i+1]*cost) ) 
            curMoney = lastMoney - position*test_data.close[i+1]
            print("long at %f at %s %s, profit is %f" %(test_data.close[i+1],test_data.date[i+1],test_data.time[i+1],profit))
            long_profit.append(profit)
            last_date.append(test_data.date[i+1])
            if(profit > 0):
                positive_profit.append(profit)
            else:
                negative_profit.append(profit)
    
    return_rate.append(curMoney/initialMoney-1)
    print ("收益率为 %f" %(curMoney/initialMoney-1) )
    if len(positive_profit) != 0:
        print ("盈利数目 %d, 平均每笔盈利 %f"%(len(positive_profit),sum(positive_profit)/(initialMoney*len(positive_profit)) ))
    else:
        print ("盈利数目 0")
    if len(negative_profit) != 0:
        print ("亏损数目 %d, 平均每笔亏损 %f"%(len(negative_profit),sum(negative_profit)/(initialMoney*len(negative_profit)) ))
    else:
        print ("亏损数目 0")
    print ("多仓平均每笔盈利 %f"%(sum(long_profit)/(initialMoney*len(long_profit)) ))
    print ("空仓平均每笔盈利 %f"%(sum(short_profit)/(initialMoney*len(short_profit)) ))
    
print (np.mean(return_rate))