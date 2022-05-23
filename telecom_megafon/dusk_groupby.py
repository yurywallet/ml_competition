# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:39:44 2018

@author: Yury
"""

import pandas as pd
df=pd.DataFrame()
df['id']=[1,1,1,2]
df['g']=[1,2,1,1]
df['a']=[1,2,100,10]
df['b']=[100,0,1,20]

#create aggregated dataframe
#temp=df.groupby(['id','g'])['a'].sum()
#temp=pd.DataFrame(temp)
#temp.columns=['sum']
#temp['drop']=temp.index
#temp=temp.reset_index(drop=True)
#temp[['id','g']] = temp['drop'].apply(pd.Series)
#temp.drop(['drop'], axis=1, inplace=True)

import dask.dataframe as dd
ddf=dd.from_pandas(df,npartitions=2)


def ag_two_dask(db, res, t_key, pr,col, m):
    
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

    res=dd.merge(res, temp.to_frame())

    return res

ttemp = ddf.groupby(['id','g'])['a'].mean().to_frame()
ttemp = ttemp.drop('a', axis=1)
for z in range(1,6):
    ttemp=ag_two_dask(ddf, ttemp,['id','g'], 'agg_','a', z)

t=ddf.groupby(['id','g'])['a'].sum()
t.name='sum'



#t=t.compute()
tt=ddf.groupby(['id','g'])['a'].mean()
tt.name='mean'
#t.divisions
#tt=tt.compute()

ttemp=dd.merge( t.to_frame(), tt.to_frame())

#ttemp=dd.concat([ttemp,t,tt], axis=1)
#
#ind=np.array(temp.index)
#indx=dd.from_pandas(pd.Series(ind, index=ind), npartitions=ttemp.npartitions)
#
#ttemp = ttemp.assign(ids=indx)

ttemp=ttemp.compute()

ttemp.compute().to_csv('../data/bs_avg_kpi_0.csv', index=True)