# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:39:47 2023

@author: 11582
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor

data=pd.read_csv('alll.csv')
data=data.query('accuracy=="1/1"&status=="Accepted"')#只要用accuracy是1/1，status是accepted的数据，为了控制变量。

languages=['C++','C','Python','Java','All']

data=data.loc[:,['problem_id','language','cpu_time','memory','code_size']]

data=data[data['language'].isin(languages)]#筛选数据只剩下需要的4种语言

def highlight_greaterthan(s, threshold, color='orange'):
    return 'background-color: {}'.format(color) if s > threshold else ''


def cv(df,model):#5折交叉验证
    size=len(df)
    if size<20:
        return pd.DataFrame([[0,0,0,0,0]],columns=['code_cpu','code_memory','code+memory_cpu','code+cpu_memory','size'])
    
    cpu=np.mean(cross_val_score(model(),df[['code_size']],df['cpu_time'],scoring='r2',n_jobs=-1))
    memory=np.mean(cross_val_score(model(),df[['code_size']],df['memory'],scoring='r2',n_jobs=-1))
    cm=np.mean(cross_val_score(model(),df[['code_size','memory']],df['cpu_time'],scoring='r2',n_jobs=-1))
    cc=np.mean(cross_val_score(model(),df[['code_size','cpu_time']],df['memory'],scoring='r2',n_jobs=-1))
    return pd.DataFrame([[cpu,memory,cm,cc,size]],columns=['code_cpu','code_memory','code+memory_cpu','code+cpu_memory','size'])


#data=data[(data.iloc[:,2:]>0).all(1)]#过滤掉CPU memory codesize为0的部分



cv_out_linear=data.groupby(['language','problem_id']).apply(lambda df:cv(df,LinearRegression))
cv_out_linear[(cv_out_linear.iloc[:,:-1]>0.5).any(1)].style.applymap(highlight_greaterthan, threshold=0.5).to_excel('linear_0.5_with0.xlsx')

cv_out_mlp=data.groupby(['language','problem_id']).apply(lambda df:cv(df,MLPRegressor))
cv_out_mlp[(cv_out_mlp.iloc[:,:-1]>0.5).any(1)].style.applymap(highlight_greaterthan, threshold=0.5).to_excel('mlp_0.5_with0.xlsx')

cv_out_rf=data.groupby(['language','problem_id']).apply(lambda df:cv(df,RandomForestRegressor))
cv_out_rf[(cv_out_rf.iloc[:,:-1]>0.5).any(1)].style.applymap(highlight_greaterthan, threshold=0.5).to_excel('rf_0.5_with0.xlsx')

cv_out_gbm=data.groupby(['language','problem_id']).apply(lambda df:cv(df,LGBMRegressor))
cv_out_gbm[(cv_out_gbm.iloc[:,:-1]>0.5).any(1)].style.applymap(highlight_greaterthan, threshold=0.5).to_excel('gbm_0.5_with0.xlsx')

cv_out_gbdt=data.groupby(['language','problem_id']).apply(lambda df:cv(df,GradientBoostingRegressor))
cv_out_gbdt[(cv_out_gbdt.iloc[:,:-1]>0.5).any(1)].style.applymap(highlight_greaterthan, threshold=0.5).to_excel('gbdt_0.5_with0.xlsx')
