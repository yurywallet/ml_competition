# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:48:42 2018

@author: Yury
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 22:38:15 2018

@author: Yury
"""

import os
#if kag==0:
#    os.environ['MKL_NUM_THREADS'] = '2' #for core i5 5200
#    os.environ['OMP_NUM_THREADS'] = '2' #for core i5 5200
#else:
#    os.environ['MKL_NUM_THREADS'] = '4' #for core i5 5200
#    os.environ['OMP_NUM_THREADS'] = '4' #for core i5 5200
    

import numpy as np
import pandas as pd
import time
st=time.clock()

import matplotlib.pyplot as plt
seed=293423
np.random.seed(seed)

# Set figure width to 12 and height to 9
fig_size=[12,9]
fig_size[0] = 14
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size



import gc

def memory_reduce(dataset):
    for col in list(dataset.select_dtypes(include=['int']).columns):
        if ((np.max(dataset[col]) <= 127) and(np.min(dataset[col]) >= -128)):
            dataset[col] = dataset[col].astype(np.int8)
        elif ((np.max(dataset[col]) <= 32767) and(np.min(dataset[col]) >= -32768)):
            dataset[col] = dataset[col].astype(np.int16)
        elif ((np.max(dataset[col]) <= 2147483647) and(np.min(dataset[col]) >= -2147483648)):
            dataset[col] = dataset[col].astype(np.int32)
    for col in list(dataset.select_dtypes(include=['float']).columns):
        dataset[col] = dataset[col].astype(np.float32)
    return dataset

def merge_l(a, b, c):
    ind=a.index
    a=a.merge(b, on=c, how='left')
    a.index=ind
    return a

#df_train = pd.read_csv('../data/train.csv',  engine="python") #index_col='zone_id'
#df_test = pd.read_csv('../data/test.csv',  engine="python") #index_col='zone_id'

sub_tr=pd.read_csv('../data/train/subs_csi_train.csv', sep=';', decimal=",",  engine="python")
sub_te=pd.read_csv('../data/test/subs_csi_test.csv',  sep=';', decimal=",", engine="python")

i='SK_ID'
i_tr=sub_tr[i]
i_te=sub_te[i]

'''target'''
tar=sub_tr['CSI']
sub_tr['CSI'].value_counts()
print('Percentage of 1 is %3f %%'%(100*sum(tar)/tar.shape[0]))

plt.hist(tar)
plt.ylabel('num')
plt.xlabel('$tar$')

''''''
#c='CONTACT_DATE'
#sub_te[c+'_y']='.2018'
#sub_te[c+'_d'] = sub_te[c].astype(str)+sub_te[c+'_y'].astype(str)
#sub_te[c+'_d']=pd.to_datetime(sub_te[c+'_d'], format="%d.%m.%Y")
#sub_te.drop([c+'_y'],inplace=True, axis=1)
#
#sub_tr[c+'_y']='.2018'
#sub_tr[c+'_d'] = sub_tr[c].astype(str)+sub_tr[c+'_y'].astype(str)
#sub_tr[c+'_d']=pd.to_datetime(sub_tr[c+'_d'], format="%d.%m.%Y")
#sub_tr.drop([c+'_y'],inplace=True, axis=1)

'''train_test'''
tr=pd.DataFrame()
tr[i]=i_tr
te=pd.DataFrame()
te[i]=i_te


def ag_one(db, res, t_key,  pr, col, agr):
    temp=db.groupby(t_key)[col].agg(agr)
    temp=temp.add_prefix(pr+col+'_')
    temp[t_key]=temp.index
    res=merge_l(res,temp,t_key)
    res.fillna(value=0, inplace=True)
    return res

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

''' features'''
def features_db(db_tr,db_te):
    
    f_tr=pd.read_csv('../data/train/subs_features_train.csv', sep=';', decimal=",",  engine="python")
    f_te=pd.read_csv('../data/test/subs_features_test.csv',  sep=';', decimal=",", engine="python")
    
    
#    f_te.loc[f_te[i]==104]
    
    #comcat 1-34
    cat_cols = [col for col in f_tr.columns if 'COM_CAT' in col]
    #fill nan as mode for user
    for col in cat_cols:
        f_te[col]=f_te.groupby([i])[col].transform(lambda x: x.fillna(x.mode()))
        f_tr[col]=f_tr.groupby([i])[col].transform(lambda x: x.fillna(x.mode()))
    #    f_te[col] = f_te.groupby([i])[col].transform(lambda x: x.fillna(x.mean()))
    
    
    #finances = 
    fin=['REVENUE', 'ITC','VAS', 'RENT_CHANNEL', 'ROAM', 'COST']
    #доли роуминг, 
    # first(), last() 
#    df=f_te.head(100)
    
    #new features
    def new_feat(db):
        fs=['ITC', 'VAS', 'RENT_CHANNEL', 'ROAM', 'COST']
         
        for c in fs:
            db[c+'2rev']=db[c]/db['REVENUE']
            db[c+'2rev'].fillna(value=0, inplace=True)   
        
        fs=['ITC', 'VAS', 'RENT_CHANNEL', 'ROAM']
         
        for c in fs:
            db[c+'2cost']=db[c]/db['COST']
            db[c+'2cost'].fillna(value=0, inplace=True) 
        
        return db
    
    f_te=new_feat(f_te)
    f_tr=new_feat(f_tr)
    
    
    col=[x for x in f_tr.columns if x not in ['SNAP_DATE', 'SK_ID']]
    
    ag=['min','max', 'mean', 'size', 'sum', 'first', 'last']
    #num_agg = {
    ##        'cnt_fut2cur_perc': ['min','max', 'mean', 'size'],
    #        'cnt_fut2cur': ['min','max', 'mean', 'size'],
    #      'CNT_INSTALMENT': ['max', 'mean', 'size'],
    #      'CNT_INSTALMENT_FUTURE': ['max', 'mean', 'size'],
    #        'MONTHS_BALANCE': ['max', 'mean', 'size'],
    #        'SK_DPD': ['max', 'mean'],
    #        'SK_DPD_DEF': ['max', 'mean']
    #        
    ##        'max_date': 'max',   # Find the max, call the result "max_date"
    ##        'min_date': 'min',
    ##        'num_days': lambda x: max(x) - min(x)
    ##        'num_days5perc': lambda x: x.quantile(0.05)
    #        
    #
    #    }
    
    
        
    #df_agg = df.groupby('SK_ID_CURR').agg(aggregations)

    
    pre='f_'
    for c in fin:
        db_tr=ag_one(f_tr, db_tr, i,  pre, c, ag)
        db_te=ag_one(f_te, db_te, i,  pre, c, ag)
        
       
    ag=['min', 'max', 'first', 'last'] # 'mode', last, first, last minus first    
    
    for c in cat_cols:
        db_tr=ag_one(f_tr, db_tr, i,  pre, c, ag)
        db_te=ag_one(f_te, db_te, i,  pre, c, ag)
    
    return db_tr, db_te

tr, te= features_db(tr,te)


''' ------------ Consumption  ------------'''

    
def consum_db(db_tr,db_te):
    
    c_tr=pd.read_csv('../data/train/subs_bs_consumption_train.csv', sep=';', decimal=",", engine="python")
    c_te=pd.read_csv('../data/test/subs_bs_consumption_test.csv',  sep=';', decimal=",", engine="python")
    
    c='SUM_MINUTES'
    
    cols=['SUM_MINUTES', 'SUM_DATA_MB', 'SUM_DATA_MIN']
    import re  
    for cc in cols: 
        c_tr[cc].fillna(value=0, inplace=True)
        c_te[cc].fillna(value=0, inplace=True)
        c_te[cc] = pd.to_numeric(c_te[cc].apply(lambda x: re.sub(',', '.', str(x))))
        c_tr[cc] = pd.to_numeric(c_tr[cc].apply(lambda x: re.sub(',', '.', str(x))))
    
    
    #1 -----------------------------------------------------------------------
    #------agregate to month
    
    ag=['mean', 'size', 'sum', 'count']    
    
    
    a_con_te=np.unique(c_te[[i, 'MON']], axis=0)
    a_con_tr=np.unique(c_tr[[i, 'MON']], axis=0)
    
    a_con_te=pd.DataFrame(a_con_te, columns=[i,'MON'])
    a_con_tr=pd.DataFrame(a_con_tr, columns=[i,'MON'])
    
#    a_con_te.dtypes
    

    

    
    
    for c in cols:
        a_con_tr=ag_two(c_tr, a_con_tr, [i,'MON'],  'c_', c, ag)
        a_con_te=ag_two(c_te, a_con_te, [i,'MON'],  'c_', c, ag)
        
    
    
    #------agregate to ID
    cols=[x for x in a_con_tr.columns if x not in ['MON', i]]
    
    pre='con_'
    ag=['min','max', 'mean', 'median', 'size', 'sum', 'first', 'last']
    
    
    
    
    for c in cols:
        db_tr=ag_one(a_con_tr, db_tr, i, pre, c, ag)
        db_te=ag_one(a_con_te, db_te, i, pre, c, ag)
    
    
    
    
    
    #2raw data statistics
    '''
    ???
    ???
    """
    '''
    gc.collect()
    return db_tr, db_te

tr, te= consum_db(tr,te)


''' ------------ Consumption data (BIG)------------'''

#cd_tr=pd.read_csv('../data/train/subs_bs_data_session_train.csv', sep=';', decimal=",", engine="python")
#cd_te=pd.read_csv('../data/test/subs_bs_data_session_test.csv',  sep=';', decimal=",", engine="python")

import dask.dataframe as dd
import re


#cd_te = dd.read_csv('../data/test/subs_bs_data_session_test.csv', sep=';', decimal=",", engine="python")
#cd_te=cd_te.compute()





#df=cd_te.head(100)

c='CELL_LAC_ID'
##stations
#temp = df.groupby(by = [i])[c].nunique().reset_index().rename(index = str, columns = {c: 'Num_stations'})
##sessions
#temp = df.groupby(by = [i])[c].count().reset_index().rename(index = str, columns = {c: 'Num_stations'})
##temp = df.groupby(by = [i])[c].size().reset_index().rename(index = str, columns = {c: 'Num_stations'})


#cols=[x for x in cd_tr.columns if x not in ['MON', i]]


#cd_tr = dd.read_csv('../data/train/subs_bs_data_session_train.csv', sep=';', decimal=",", engine="python")
#cd_tr=cd_tr.compute()
#
#'''SPLIT column by space to DAY and TIME'''
#def split_time(db, col):
#    
##    db = db.join(db[col].str.split(' ', n=1, expand = True))
##    db = db.rename(columns={0: "DAY", 1: "TIME"})
#    db[['DAY','TIME']]=db["START_TIME"].apply(lambda x: str(x).split(' ')).apply(pd.Series)
#    #df.drop([0,1], axis=1, inplace=True)
#    db['MON']=db['DAY'].apply(lambda x: str('01.')+str(x).split('.')[1])
##    db.drop([col], axis=1, inplace=True)
#    return db


#cd_te=split_time(cd_te, "START_TIME")
#import time
#st=time.clock()
#df=cd_te.head(1000)
#print (time.clock()-st)
#df=split_time(df, "START_TIME")
##df['t']=pd.to_datetime(df["START_TIME"], format="%d.%m %H:%M:%S")
#cd_te.to_csv(os.path.dirname(os.getcwd())+'\\data\\test\\subs_bs_data_session_test_1.csv', index=False)

#gc.collect()


gc.collect()



    


st=time.clock()
for j in range (0,4):
    temp = dd.read_csv('../data/train/subs_bs_data_session_train_0'+str(j)+'.csv', sep=',', decimal=",", engine="python")
    temp=temp.compute()
    temp['DATA_VOL_MB'].fillna(value=0, inplace=True)
    temp['DATA_VOL_MB'] = pd.to_numeric(temp['DATA_VOL_MB'].apply(lambda x: re.sub(',', '.', str(x))))
    if j==0:
        cd_tr=pd.DataFrame(columns=list(temp.columns))

    cd_tr=pd.concat([cd_tr,temp],axis=0)  
 
print (time.clock()-st)


st=time.clock()
for j in range (0,5):
    temp = dd.read_csv('../data/test/subs_bs_data_session_test_0'+str(j)+'.csv', sep=',', decimal=",", engine="python")
    temp=temp.compute()
    temp['DATA_VOL_MB'].fillna(value=0, inplace=True)
    temp['DATA_VOL_MB'] = pd.to_numeric(temp['DATA_VOL_MB'].apply(lambda x: re.sub(',', '.', str(x))))
    if j==0:
        cd_te=pd.DataFrame(columns=list(temp.columns))

    cd_te=pd.concat([cd_te,temp],axis=0)  
 
print (time.clock()-st)


cols=['DATA_VOL_MB']
for cc in cols: 
    cd_te[cc].fillna(value=0, inplace=True)
    cd_te[cc] = pd.to_numeric(cd_te[cc].apply(lambda x: re.sub(',', '.', str(x))))
    
    cd_tr[cc].fillna(value=0, inplace=True)
    cd_tr[cc] = pd.to_numeric(cd_tr[cc].apply(lambda x: re.sub(',', '.', str(x))))



pre='dc_'

ag=['mean', 'median', 'size', 'sum', 'first', 'last']

#cols=[x for x in cd_tr.columns if x not in [i, 'MON', "TIME", 'DAY']]



i='SK_ID'
#a_cd_te=np.unique(cd_te[[i, 'MON','DAY']], axis=0)
#a_cd_tr=np.unique(cd_tr[[i, 'MON','DAY']], axis=0)

a_cd_te=cd_te.drop_duplicates([i, 'MON','DAY'])[[i, 'MON','DAY']]
a_cd_tr=cd_tr.drop_duplicates([i, 'MON','DAY'])[[i, 'MON','DAY']]
a_cd_te.dtypes

agg_cd_te=a_cd_te.drop_duplicates([i, 'MON'])[[i, 'MON']]
agg_cd_tr=a_cd_tr.drop_duplicates([i, 'MON'])[[i, 'MON']]

cd_tr['DAY']= cd_tr['DAY'].astype(str)
cd_tr['MON']= cd_tr['MON'].astype(str)

cd_te['DAY']= cd_te['DAY'].astype(str)
cd_te['MON']= cd_te['MON'].astype(str)

#agg to day
ag=['mean', 'median', 'size', 'sum']
cols=['DATA_VOL_MB']
for c in cols:
    a_cd_tr=ag_two(cd_tr, a_cd_tr, [i, 'MON','DAY'], 'dc_', c, ag)
    a_cd_te=ag_two(cd_te, a_cd_te, [i,'MON','DAY'], 'dc_', c, ag)



#c_tr.dtypes


#from pandasql import sqldf
#q = """SELECT DISTINCT MON FROM a_cd_te;"""
#pysqldf = lambda q: sqldf(q, globals())
#a_cd_te_q = pysqldf(q)


cols=[x for x in a_cd_tr.columns if x not in [i, 'MON', 'DAY']]

#
ag=['mean', 'median', 'size', 'sum', 'first', 'last']
for c in cols:
    tr=ag_one(a_cd_tr, tr, i, 'dc_', c, ag)
    te=ag_one(a_cd_te, te, i, 'dc_', c, ag)





#-----------------------Classification----------------
from sklearn.metrics import roc_auc_score


from sklearn.linear_model import RidgeClassifier
t=time.clock()        
classifier = RidgeClassifier()
classifier.fit(tr, tar)
pred=classifier.predict(tr)
print('ROC AUC',roc_auc_score(tar,pred))



from sklearn.linear_model import LogisticRegression
t=time.clock()        
classifier = LogisticRegression()
classifier.fit(tr, tar)
pred=classifier.predict_proba(tr)[0:,1]
print('ROC AUC',roc_auc_score(tar,pred))

import xgboost as xgb
classifier=xgb.XGBClassifier()
classifier.fit(tr, tar)
pred=classifier.predict_proba(tr)[0:,1]
print('ROC AUC',roc_auc_score(tar,pred))


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 100)
classifier.fit(tr, tar)
pred=classifier.predict_proba(tr)[0:,1]
print('ROC AUC',roc_auc_score(tar,pred))


from sklearn.svm import SVC
classifier= SVC(kernel='rbf',random_state=0)
classifier.fit(tr, tar)
pred=classifier.predict(tr)
print('ROC AUC',roc_auc_score(tar,pred))

from sklearn.ensemble import ExtraTreesClassifier
classifier = ExtraTreesClassifier(n_estimators = 150, criterion = 'entropy', random_state = 100)
classifier.fit(tr, tar)
pred=classifier.predict(tr)
print('ROC AUC',roc_auc_score(tar,pred))

from lightgbm             import LGBMClassifier
from xgboost              import XGBClassifier
rs=946437 

n=3000
es=500
clfs = {
        'xgb_cl':XGBClassifier(seed=111, 
                               n_estimators=n,
                               nthread=3,
                               max_depth=6, 
                               learning_rate=0.01,
                               objective='binary:logistic',
#                               eval_metric='auc',
                               random_state=rs) #, n_jobs=-1
        , 
        'lgbm_cl':LGBMClassifier(n_estimators=n, silent=True, 
                                 learning_rate=0.01,
                                 boosting_type='dart',
                                 objective='binary',
                                 nthread=3,
#                                 num_leaves=-1,
#                                 colsample_bytree=0.9497036,
#                                 subsample=0.8715623,
                                 max_depth=-1,
#                                 reg_alpha=0.04,
#                                 reg_lambda=0.073
                                 random_state=rs
                                 )    
    }
folds=10
rep=1
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=121)


sam=pd.DataFrame()
sam[i]=i_tr
sam['tar']=tar

te.drop([i], axis=1, inplace=True)
tr.drop([i], axis=1, inplace=True)
#te[i]=i_te
#tr[i]=i_tr


#create balanced training set
sam_0=sam.loc[sam['tar']==0]
sam_1=sam.loc[sam['tar']==1]

sam_tr_1=tr.loc[tr[i].isin(list(sam_1[i]))]
n=656
sam_tr_0 = tr.loc[tar == 1].sample(n=n, replace=False, random_state=0)








#
#tr.drop(c,axis=1, inplace=True)

#voting
av_score=0
av_score_max=0
av_score_min=0
av_score_mean=0


df_train_pred=pd.DataFrame(index=tr.index)

pred=pd.DataFrame(index=te.index)
#df_train_pred=pd.DataFrame(index=df_test.index)
fol=0
for train_index, test_index in skf.split(tr, tar):
    
    df_valid_pred=pd.DataFrame(index=tar.iloc[test_index].index)
    av_score_f=0
    for c in list(clfs.keys()):
        classifier=clfs[c]
        
        if c in ['lgbm_cl', 'xgb_cl']:
            bst=classifier.fit(tr.iloc[train_index], np.array(tar.iloc[train_index]),
                            eval_set=[ (tr.iloc[train_index], np.array(tar.iloc[train_index])),
                                    (tr.iloc[test_index], np.array(tar.iloc[test_index]))], 
                            eval_metric= 'auc', 
                            verbose= 200, 
                            early_stopping_rounds= es)
            
            if c=='lgbm_cl':
                y_pred = bst.predict_proba(tr.iloc[test_index], num_iteration=bst.best_iteration_)
                '''pred'''
                y_tar=bst.predict_proba(te, num_iteration=bst.best_iteration_)[:,1]
            else:
                #xgboost
                y_pred = bst.predict_proba(tr.iloc[test_index], ntree_limit=bst.best_ntree_limit)
                '''pred'''
                y_tar = bst.predict_proba(te, ntree_limit=bst.best_ntree_limit)[:,1]
        else:
            bst=classifier.fit(tr.iloc[train_index], np.array(tar.iloc[train_index]))
            y_pred = bst.predict_proba(tr.iloc[test_index])
            '''pred'''
            y_tar = bst.predict_proba(te)[:,1]
        
        pred[c+'_'+str(fol)]=y_tar
        df_valid_pred[c+'_'+str(2)]=y_pred[:,1]
        target=tar.iloc[test_index]
        
        
        av_score+=roc_auc_score(tar.iloc[test_index], y_pred[:,1])
        av_score_f+=roc_auc_score(tar.iloc[test_index], y_pred[:,1])
        
        print('---------------------------------------')
        print(fol, " Qual ", c," : " , roc_auc_score(tar.iloc[test_index], y_pred[:,1]))
        print('---------------------------------------')
        
    #pred valid
    y_pred_v=df_valid_pred.mean(axis=1)
    print(fol, " Qual mean : " , roc_auc_score(target, y_pred_v))
    av_score_mean+=roc_auc_score(target, y_pred_v)
    y_pred_v=df_valid_pred.min(axis=1)
    print(fol, " Qual min : " , roc_auc_score(target, y_pred_v))
    av_score_min+=roc_auc_score(target, y_pred_v)
    y_pred_v=df_valid_pred.max(axis=1)
    print(fol, " Qual max : " , roc_auc_score(target, y_pred_v))
    av_score_max+=roc_auc_score(target, y_pred_v)
    #predict train
        
    #predict test
           
    print(fol, "_Av_Qual: " , av_score_f/2)
    fol+=1
    
    if c=='lgbm_cl':
        a=classifier.feature_importances_
        a=pd.DataFrame(a, index=tr.columns)
print('---------------------------------------')
print(c, "Av_Qual: " , av_score/(folds*rep*2))
print("Av_Qual_mean: " , av_score_mean/(folds*rep))
print("Av_Qual_min: " , av_score_min/(folds*rep))
print("Av_Qual_max: " , av_score_max/(folds*rep))    

c=[x for x in tr.columns if x not in te.columns]
print(c)
#------------------regression-----------------------

#
#from sklearn.linear_model import Ridge
#params={'alpha':[0.1,0.001, 0.0005], 'fit_intercept':[False, True]}
#model=Ridge()
#
#from sklearn.linear_model import BayesianRidge
#params={'alpha_1':[0.001, 0.0005], 'fit_intercept':[False, True]}
#model=BayesianRidge()
#
#









#---------------------------------------------------------------------------
import datetime
subm=pd.DataFrame()
subm['fin']=pred.max(axis=1)

td=datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")


subm.to_csv(os.path.dirname(os.getcwd())+'\\subm\\submission'+ '_'+ td+'.csv', index=False, header=False)


from save_zip import save_src_to_zip

save_src_to_zip(os.path.dirname(os.getcwd())+'\\src_zip\\',  exclude_folders = ['__pycache__'], dname="src", td=td)


