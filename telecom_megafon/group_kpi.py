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
else:
    os.environ['MKL_NUM_THREADS'] = '4' 
    os.environ['OMP_NUM_THREADS'] = '4' 
    

import numpy as np
import pandas as pd
import time
import gc
import dask.dataframe as dd

def merge_l(a, b, c):
    ind=a.index
    a=a.merge(b, on=c, how='left')
    a.index=ind
    return a

def ag_two(db, res, t_key,  pr, col, agr):
    temp=db.groupby(t_key)[col].agg(agr)
    temp=temp.add_prefix(pr+col+'_')
    temp['drop']=temp.index
    temp=temp.reset_index(drop=True)
    temp[t_key] = temp['drop'].apply(pd.Series)
    temp.drop(['drop'], axis=1, inplace=True)
    res=merge_l(res,temp,t_key)
    res.fillna(value=0, inplace=True)
    return res


def ag_two_dask(db, res, t_key,  pr, col, m):
    
    if m==1:
        temp = db.groupby(t_key)[cols].mean()
        temp.name=col+'_mean'
    if m==2:    
        temp=db.groupby(t_key)[col].std()
        temp.name=col+'_mean'
    if m==3:    
        temp=db.groupby(t_key)[col].sum()   
        temp.name=col+'_mean'
    if m==4:    
        temp=db.groupby(t_key)[col].count()   
        temp.name=col+'_mean'
    if m==5:    
        temp=db.groupby(t_key)[col].nunique()   
        temp.name=col+'_mean'

        
    temp=temp.add_prefix(pr+col+'_')
    temp['drop']=temp.index
    temp=temp.reset_index(drop=True)
    temp[t_key] = temp['drop'].apply(pd.Series)
    temp.drop(['drop'], axis=1, inplace=True)

    res=res.merge(temp, on=t_key)  
    
    res.fillna(value=0, inplace=True)
    return res

cols=['CELL_AVAILABILITY_2G', 
'CELL_AVAILABILITY_3G', 
'CELL_AVAILABILITY_4G', 
'CSSR_2G', 
'CSSR_3G', 
'ERAB_PS_BLOCKING_RATE_LTE', 
'ERAB_PS_DROP_RATE_LTE', 
'HSPDSCH_CODE_UTIL_3G', 
'NODEB_CNBAP_LOAD_HARDWARE', 
'PART_CQI_QPSK_LTE', 
'PART_MCS_QPSK_LTE', 
'PROC_LOAD_3G', 
'PSSR_2G', 
'PSSR_3G', 
'PSSR_LTE', 
'RAB_CS_BLOCKING_RATE_3G', 
'RAB_CS_DROP_RATE_3G', 
'RAB_PS_BLOCKING_RATE_3G', 
'RAB_PS_DROP_RATE_3G', 
'RBU_AVAIL_DL', 
'RBU_AVAIL_UL', 
'RBU_OTHER_DL', 
'RBU_OTHER_UL', 
'RBU_OWN_DL', 
'RBU_OWN_UL', 
'RRC_BLOCKING_RATE_3G', 
'RRC_BLOCKING_RATE_LTE', 
'RTWP_3G', 
'SHO_FACTOR', 
'TBF_DROP_RATE_2G', 
'TCH_DROP_RATE_2G', 
'UTIL_BRD_CPU_3G', 
'UTIL_CE_DL_3G', 
'UTIL_CE_HW_DL_3G', 
'UTIL_CE_UL_3G', 
'UTIL_SUBUNITS_3G', 
'UL_VOLUME_LTE', 
'DL_VOLUME_LTE', 
'TOTAL_DL_VOLUME_3G', 
'TOTAL_UL_VOLUME_3G'] 

c='CELL_AVAILABILITY_2G'
ucols=['T_DATE', 'CELL_LAC_ID']
ucolss=ucols+[c]
res = dd.read_csv('../data/bs_avg_kpi.csv',  sep=';', decimal=",", engine="python", usecols=ucolss)
#res=res.drop_duplicates(ucols)
res = res.groupby(ucols)['CELL_AVAILABILITY_2G'].mean()
res = res.groupby(ucols)['CELL_AVAILABILITY_2G'].sum()

res=res.compute()

 
#.compute()

#res=db.drop_duplicates() #.compute()
#
#ucolss=ucols+[cols[0]]
#db = dd.read_csv('../data/bs_avg_kpi.csv',  sep=';', decimal=",", engine="python", usecols=ucolss)
##res=ag_two_dask(db, res, ucols,  'kpi', cols[0],1)
#
#res1=res.compute()
#
#db=db.compute()
#res=pd.Dataframe()
#res1=res1.drop_duplicates(ucols)[ucols]

def load_group(res, c):
    ucolss=ucols+[c]
    db = dd.read_csv('../data/bs_avg_kpi.csv',  sep=';', decimal=",", engine="python", usecols=ucolss)
    db=db.compute()

    ag=['min','max', 'mean', 'size', 'sum', 'first', 'last']

    res=ag_two(db, res, ucols,  'akpi_', c, ag)
        
    return res

for c in cols:
     res = load_group(res, c) 




#for i in range (2,5):
#    st=time.clock()
#    temp =cd_tr.iloc[1000000*i:1000000*(i+1),:]
#    temp=split_time(temp,'START_TIME' )
#    
#   
res.to_csv('bs_avg_kpi_0.csv', index=False)

#    print (time.clock()-st)
