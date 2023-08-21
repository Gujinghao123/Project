# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:29:03 2023

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


def cv(df,model,parameters={}):#5折交叉验证
    size=len(df)
    if size<20:
        return pd.DataFrame([[0,0,0,0,0]],columns=['code_cpu','code_memory','code+memory_cpu','code+cpu_memory','size'])
    result=[]
    for pair in [
        [['code_size'],'cpu_time'],
        [['code_size'],'memory'],
        [['code_size','memory'],'cpu_time'],
        [['code_size','cpu_time'],'memory']
        ]:
        gs=GridSearchCV(model(),param_grid=parameters,scoring='r2',n_jobs=4,pre_dispatch=4)
        gs.fit(df[pair[0]],df[pair[1]])
        result.append(gs.best_score_)
    result.append(size)
    return pd.DataFrame([result],columns=['code_cpu','code_memory','code+memory_cpu','code+cpu_memory','size'])



#data=data[(data.iloc[:,2:]>0).all(1)]#过滤掉CPU memory codesize为0的部分

linear_param={'fit_intercept':[True,False]}

mlp_param = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50)],
    'activation': ['tanh', 'relu'],  
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

rf_param = {
    'n_estimators': [10, 50],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [2,4,6,8,10],
    'bootstrap': [True, False]
}

gbm_param = {
    'num_leaves': [31, 127],
    'reg_alpha': [0.1, 0.5],
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
    }

gbdt_param = {'n_estimators':[100, 200, 300, 400, 500], 
            'learning_rate': [0.1, 0.05, 0.01],
            'max_depth':[4, 6, 8], 
            'min_samples_leaf':[3, 5, 9, 14], 
            'max_features':[0.1, 0.3, 1.0]}

#cv_out_linear=data.groupby(['language','problem_id']).apply(lambda df:cv(df,LinearRegression,linear_param))
#cv_out_linear[(cv_out_linear.iloc[:,:-1]>0.5).any(1)].style.applymap(highlight_greaterthan, threshold=0.5).to_excel('linear_0.5_best.xlsx')

#cv_out_mlp=data.groupby(['language','problem_id']).apply(lambda df:cv(df,MLPRegressor,mlp_param))
#cv_out_mlp[(cv_out_mlp.iloc[:,:-1]>0.5).any(1)].style.applymap(highlight_greaterthan, threshold=0.5).to_excel('mlp_0.5_best.xlsx')

#cv_out_rf=data.groupby(['language','problem_id']).apply(lambda df:cv(df,RandomForestRegressor,rf_param))
#cv_out_rf[(cv_out_rf.iloc[:,:-1]>0.5).any(1)].style.applymap(highlight_greaterthan, threshold=0.5).to_excel('rf_0.5_best.xlsx')

#cv_out_gbm=data.groupby(['language','problem_id']).apply(lambda df:cv(df,LGBMRegressor,gbm_param))
#cv_out_gbm[(cv_out_gbm.iloc[:,:-1]>0.5).any(1)].style.applymap(highlight_greaterthan, threshold=0.5).to_excel('gbm_0.5_best.xlsx')

cv_out_gbdt=data.groupby(['language','problem_id']).apply(lambda df:cv(df,GradientBoostingRegressor,gbdt_param))
cv_out_gbdt[(cv_out_gbdt.iloc[:,:-1]>0.5).any(1)].style.applymap(highlight_greaterthan, threshold=0.5).to_excel('gbdt_0.5_best.xlsx')

