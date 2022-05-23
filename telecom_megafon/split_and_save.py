# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 13:43:47 2018

@author: Yury
"""

import os
kag=0
if kag==0:
    os.environ['MKL_NUM_THREADS'] = '3' #for core i5 5200
    os.environ['OMP_NUM_THREADS'] = '3' #for core i5 5200
#else:
#    os.environ['MKL_NUM_THREADS'] = '4' #for core i5 5200
#    os.environ['OMP_NUM_THREADS'] = '4' #for core i5 5200
    

import numpy as np
import pandas as pd
import time
import gc
import dask.dataframe as dd


'''SPLIT column by space to DAY and TIME'''
def split_time(db):
    
#    db = db.join(db[col].str.split(' ', n=1, expand = True))
#    db = db.rename(columns={0: "DAY", 1: "TIME"})
#    db[['DAY','TIME']]=db["START_TIME"].apply(lambda x: str(x).split(' ')).apply(pd.Series)
#    #df.drop([0,1], axis=1, inplace=True)
#    db['MON']=db['DAY'].apply(lambda x: str('01.')+str(x).split('.')[1])
##    db.drop([col], axis=1, inplace=True)
    
    db['t']=pd.to_datetime(db["START_TIME"], format="%d.%m %H:%M:%S")
    db['DAY']=db['t'].dt.strftime("%d.%m") #object
    db['TIME']=db['t'].dt.strftime("%H:%M:%S") #object
    db['MON']=db['t'].dt.strftime("01.%m") #object
    db.drop(['t'], axis=1, inplace=True)
    return db



folds=['train',]
for fold in folds:
    bs_tr = dd.read_csv('../data/'+fold+'/'+'subs_bs_voice_session_'+fold+'.csv', sep=';', decimal=",", engine="python")
    bs_tr=bs_tr.compute()

    bs_tr=split_time(bs_tr)
    bs_tr.to_csv(os.path.dirname(os.getcwd())+'\\data\\'+fold+'\\'+'subs_bs_voice_session_'+fold+'_0.csv', index=False)

#st=time.clock()
#
#gc.collect()
#
#
#
#for i in range (2,5):
#    st=time.clock()
#    temp =cd_tr.iloc[1000000*i:1000000*(i+1),:]
#    temp=split_time(temp,'START_TIME' )
#    
#   
#    temp.to_csv(os.path.dirname(os.getcwd())+'\\data\\bs_avg_kpi_0'+str(i)+'.csv', index=False)
#
#    print (time.clock()-st)
