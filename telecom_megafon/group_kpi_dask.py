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

# find only cells that are in other data

c='CELL_LAC_ID'
#subs_bs_consumption 
cell_id_con=[]
for col in ['train', 'test']:
    rez = dd.read_csv('../data/'+col+'/subs_bs_consumption_'+col+'.csv',  sep=';', decimal=",", engine="python", usecols=[c])
    rez=rez.drop_duplicates().compute()
    ids=list(rez[c])
    #ids=list(rez['CELL_LAC_ID'].drop_duplicates())
    cell_id_con=list(set(ids+cell_id_con))


#subs_bs_data_session
cell_id_dat=[]
for col in ['train', 'test']:
    if col=='train': 
        i=4
    else:
        i=5
    for j in range(i):
        rez = dd.read_csv('../data/'+col+'/subs_bs_data_session_'+col+'_0' +str(j)+'.csv',  sep=',', engine="python", usecols=[c])
        rez=rez.drop_duplicates().compute()
        ids=list(rez[c])
        #ids=list(rez['CELL_LAC_ID'].drop_duplicates())
        cell_id_dat=list(set(ids+cell_id_dat))
    
    
#subs_bs_voice_session
cell_id_voi=[]
for col in ['train', 'test']:
    rez = dd.read_csv('../data/'+col+'/subs_bs_voice_session_'+col+'_0.csv',  sep=',', engine="python", usecols=[c])
    rez=rez.drop_duplicates().compute()
    ids=list(rez[c])
    #ids=list(rez['CELL_LAC_ID'].drop_duplicates())
    cell_id_voi=list(set(ids+cell_id_voi))


cells=list(set(cell_id_dat+cell_id_con+cell_id_voi))


def ag_two_dask(db, res, t_key, pr,col, m):
    
#    t_key=ucols
#    col=c
#    pr='1_'
    if m==1:
        temp = db.groupby(t_key)[col].mean().fillna(0)
        temp.name=pr+col+'_mean'
    if m==2:    
        temp=db.groupby(t_key)[col].std().fillna(0)
        temp.name=pr+col+'_std'
    if m==3:    
        temp=db.groupby(t_key)[col].sum().fillna(0)   
        temp.name=pr+col+'_sum'
    if m==4:    
        temp=db.groupby(t_key)[col].count().fillna(0)   
        temp.name=pr+col+'_count'
    if m==5:    
        temp=db.groupby(t_key)[col].nunique().fillna(0)   
        temp.name=pr+col+'_nunique'
#    res=rez
    res = res.repartition(npartitions=200)
    res = res.reset_index(drop=True)
    temp = temp.repartition(npartitions=200)
    temp = temp.reset_index(drop=True)
#    res.divisions
#    temp.divisions
#    res = res.assign(col = temp[col])
#    temp
#    res=dd.merge(res, temp.to_frame())
    res=dd.concat([res,temp],axis=1,interleave_partitions=True)
    
    return res

cols=[ 
'CELL_AVAILABILITY_2G', 
'CELL_AVAILABILITY_3G', 
'CELL_AVAILABILITY_4G', 
'CSSR_2G', 
'CSSR_3G', 
'ERAB_PS_BLOCKING_RATE_LTE', 
'ERAB_PS_DROP_RATE_LTE', 
'PSSR_2G', 
'PSSR_3G', 
'PSSR_LTE', 
'RAB_CS_BLOCKING_RATE_3G', 
'RAB_CS_DROP_RATE_3G', 
'RAB_PS_BLOCKING_RATE_3G', 
'RAB_PS_DROP_RATE_3G', 
'RBU_AVAIL_DL', 
'RBU_AVAIL_UL', 
'RRC_BLOCKING_RATE_3G', 
'RRC_BLOCKING_RATE_LTE', 
'TBF_DROP_RATE_2G', 
'TCH_DROP_RATE_2G'] 

#c='CELL_AVAILABILITY_2G'
ucols=['T_DATE', 'CELL_LAC_ID']

rez = dd.read_csv('../data/bs_avg_kpi.csv',  sep=';', decimal=",", engine="python", usecols=ucols)
rez=rez.drop_duplicates()
rez=rez[rez['CELL_LAC_ID'].isin(cells)]

#h=rez.head()
''' SKIP T_date to reduce the amount of value'''
#
#cols=[ 
#'CELL_AVAILABILITY_2G', 
#'CELL_AVAILABILITY_3G']


#def load_group(res, c):
#    ucolss=ucols+[c]
#    db = dd.read_csv('../data/bs_avg_kpi.csv',  sep=';', decimal=",", engine="python", usecols=ucolss)
#    db=db[db['CELL_LAC_ID'].isin(cells)]
##    len(db) 
#    for z in range(1,6):
#        res=ag_two_dask(db, res, ucols,  'akpi_', c, z)   
#    return res
#
#st=time.clock()
#for c in cols:
#    print (c)
#    rez = load_group(rez, c) 
#    print (time.clock()-st)


st=time.clock()
colz=ucols+cols
df = dd.read_csv('../data/bs_avg_kpi.csv',  sep=';', decimal=",", engine="python", usecols=colz)
df=df[df['CELL_LAC_ID'].isin(cells)]
for c in cols:
    ucolss=ucols+[c]
    for z in range(1,6):
        rez=ag_two_dask(df[ucolss], rez, ucols,  'akpi_', c, z) 
print (time.clock()-st)

gc.collect()
#split into 5 parts

def sv(rez, cells):
    for z in range(0,20):
        st=time.clock()
        cells_1=cells[z*25000:(z+1)*25000]
        res=rez[rez['CELL_LAC_ID'].isin(cells_1)].repartition(npartitions=1).reset_index(drop=True)
        print ('start saving')
        res.compute().to_csv(os.path.dirname(os.getcwd())+'\\data\\bs_avg_kpi_0'+str(z)+'.csv', index=True)
        print (time.clock()-st)
 
sv(rez, cells)

#    print (time.clock()-st)
