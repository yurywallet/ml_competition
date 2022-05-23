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
from math import sin, cos, sqrt, atan2, radians
#from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors


from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

import pickle




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

def merge_l(a, b, col):
    ind=a.index
    a=a.merge(b, on=col, how='left')
    a.index=ind
    return a

#import geopy.distance
#
##coords_1 = (52.2296756, 21.0122287)
##coords_2 = (52.406374, 16.9251681)
##
##print (geopy.distance.geodesic(coords_1, coords_2).km)
#
#def distance(x,y):
#    return geopy.distance.geodesic(x, y).km




def distance(x,y):
    R = 6373.0 # радиус земли в километрах
    """
    Параметры
    ----------
    x : tuple, широта и долгота первой геокоординаты 
    y : tuple, широта и долгота второй геокоординаты 
    
    Результат
    ----------
    result : дистанция в километрах между двумя геокоординатами
    """
    lat_a, long_a, lat_b, long_b = map(radians, [*x,*y])    
    dlon = long_b - long_a
    dlat = lat_b - lat_a
    a = sin(dlat/2)**2 + cos(lat_a) * cos(lat_b) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c
    

#distance(coords_1, coords_2)





df_train = pd.read_csv('../data/train.csv', index_col=0)
df_train.head()

df_train[['target']].describe()

df_train['atm_group'].value_counts() 



''' balance '''

#BAR plot



#_______________________________________________

df_test = pd.read_csv('../data/test.csv', index_col=0)
df_test.head()





import matplotlib.pyplot as plt
def histog(db, apl, off, label_1, label_2, st=10, bins=1):
    
    if db[apl].dtype!='O':
        mi=min(min(db[apl]),min(db[off]))
        ma=max(max(db[apl]),max(db[off]))+st
        if bins==1:
            bins=range(mi, ma+st,st) -np.ones(len(range(mi, ma+st,st)))*st/2
        else:
            bins=None
    else:
        bins=None
    fig = plt.figure(figsize=(20, 7))
    

    ax1 = fig.add_subplot(1, 4, 1)   

    ax1.hist(db[apl], histtype='bar', 
             color='dodgerblue',
             bins=bins, label=label_1,
             align='mid') #, normed=True
    ax1.hist(db[off], label=label_2,
             color='darkslateblue',alpha=0.75, 
             bins= bins,
             align='mid' )#, normed=True
    if db[apl].dtype!='O':
        ax1.set_xlim([mi-st, ma])
    ax1.legend()
    #plt.xticks([st*i for i in range(0, 21)])

    
    
    #look in scorecards
    

    for i in range(1,3):
        
        if i==1:
            c=True
        elif i==2:
            c=False

        df=db[db['isTrain']==c]
        ax = fig.add_subplot(1, 4, i+1, sharey=ax1) #
        ax.hist(df[apl], histtype='bar', 
                color='dodgerblue',
                bins=bins, label=label_1,
                align='mid') #, normed=True
        ax.hist(df[off], label=label_2,alpha=0.8,
                color='darkslateblue',
                bins=bins,
                align='mid') #, normed=True
        ax.text(0.7,0.7, str(c),horizontalalignment='center',
            verticalalignment='center',
               fontsize=26, ha='center',
               transform=ax.transAxes,
               color='darkslateblue')
        if db[apl].dtype!='O':
            ax.set_xlim([mi-st, ma])
        ax.legend()

    plt.tight_layout()
    plt.show()


df_train['isTrain'] = True
df_test['isTrain'] = False
print('missing in train {} missig in test {}'.format(df_train[df_train['lat'].isna()].shape[0], df_test[df_test['lat'].isna()].shape[0]))


b=df_test[df_test['address_rus'].str.contains('Домодедово')==True]

b=df_test[df_test['lat'].isna()][['address', 'lat', 'long']]
b.to_csv('lost_coord.csv', sep=';')

b=df_train.sort_values(['atm_group','id'], ascending=[1,1]) 

'''???????'''
#Drop from train all missing
y=0
if y==1:
    x=df_train[df_train['lat'].isna()]['address']
    xx=df_test[df_test['lat'].isna()]['address']
    x=[i for i in x if i not in xx ]

    df_train_c=df_train[~df_train['address'].isin(x)]



    X = df_train_c.append(df_test)
    x_ind=X.index
    
else:
    
    X = df_train.append(df_test)
    x_ind=X.index


X['atm_group']=X['atm_group'].astype(str)
dg='dodgerblue'
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

fig, axes = plt.subplots(figsize=(18,8),nrows=1, ncols=3, sharey=True)
ax1 = axes[0]
ax2 = axes[1]
ax3 = axes[2]
X['atm_group'].value_counts().plot(ax=ax1, kind='bar', color=dg, label ='u')

X[X['isTrain']==True]['atm_group'].value_counts().plot(ax=ax2, kind='bar', color=dg)
X[X['isTrain']==False]['atm_group'].value_counts().plot(ax=ax3, kind='bar', color=dg)
ax1.text(right, top, "ALL",         
         horizontalalignment='right',
         verticalalignment='top',
         transform=ax1.transAxes)
ax2.text(right, top, "TRAIN",         
         horizontalalignment='right',
         verticalalignment='top',
         transform=ax2.transAxes)
ax3.text(right, top, "TEST",         
         horizontalalignment='right',
         verticalalignment='top',
         transform=ax3.transAxes)
plt.tight_layout()
plt.show()





#histog(X, 'atm_group', off, 'Train', 'Test', st=10, bins=1)


a=X[X['lat'].isna()][['address','atm_group']].sort_values(by='atm_group').drop_duplicates()



cols=list(df_train.columns)

''' missing------------------------------------------------'''
#def handle_missing(dataset, col):
#    dataset[col].fillna(value="noch", inplace=True)
#    return (dataset)
#

#
#
#df_train=handle_missing(df_train,"channel_type")
#df_test=handle_missing(df_test,"channel_type")
#




import re

def a_city(s):
    
    l = s.split(' ')
    p=[]
    for j in range(len(l)):
        if len(l[j])>1:
            p.append(l[j])
    pp=str()
    for j in range(len(p)-1):       
        pp+=str(p[j]) + ' '
    
    return pp.lower().strip(), p[-1].lower().strip()


di={
    '14moscow':'moscow',
    '4/1moskva':'moscow',

    '7moskva':'moscow',
    'moskva':'moscow',
'zuevo':'orekhovo-zue', 
'zheleznogorsk':'zheleznogors', 
'zheleznodoro':'zheleznodorozhn', 
'zaozernyy':'zaozernyj', 
'yuzhno-sakhalin':'y-sakhalinsk', 
'yuzhno-sakhal':'y-sakhalinsk', 
'yuzhno-sakha':'y-sakhalinsk', 
'yu-sakhalins':'y-sakhalinsk', 
'yu-sahalinsk':'y-sakhalinsk', 
'yu.sakhalins':'y-sakhalinsk', 
'yelizovo':'elizovo', 
'yelets':'elets', 
'yekaterinburg':'ekaterinburg', 
'yalutorovsk':'jalutorovsk', 
'y.sakhalinsk':'y-sakhalinsk', 
'xtovo':'kstovo', 
'volzhsky':'volzhskiy', 
'volzhskij':'volzhskiy', 
'volno-nadezh':'volno-nadezhdin', 
'viljuchinsk':'vilyuchinsk', 
'vel':'novgoro', 
'v.pyshma':'verkhnyaya', 
'v.novgorod':'novgoro', 
'uzhno-sakhal':'y-sakhalinsk', 
'ussuriysk':'ussurijsk', 
'usolye-sibir':'usolie-sibir', 
'u-sahalinsk':'y-sakhalinsk', 
'u.-sakhalins':'y-sakhalinsk', 
'tyumen':'tumen', 
'troitskoe':'troitskoye', 
'tolyatti':'togliatti', 
'toljatti':'togliatti', 
'toglistti':'togliatti', 
'tjumen':'tumen', 
'tagil':'n.tagil', 
'tagi':'n.tagil', 
'svobodnyy':'svobodny', 
'svobodnyj':'svobodny', 
'svobodniy':'svobodny', 
'stavropol\'':'stavropol', 
'st.petersburg':'s.-petersbur', 
'st.peterburg':'s.-petersbur', 
'spassk-dalni':'dalni', 
'slob':'rybinsk', 
'shuja':'shuya', 
'shchelkovo':'schelkovo', 
'shchekino':'shekino', 
'shakhti':'shakhty', 
'serpukhov':'serpuhov', 
'serishevo':'seryshevo', 
'santk-peterb':'s.-petersbur', 
'sankt-peters':'s.-petersbur', 
'sankt-peterburg':'s.-petersbur', 
'sankt-peterb':'s.-petersbur', 
'sakhal':'y-sakhalinsk', 
's.-peterburg':'s.-petersbur', 
's.peterburg':'s.-petersbur', 
's.':'vorsino', 
'rybnoye':'rybnoe', 
'rtishevo':'rtishchevo', 
'rostov-on-do':'rostov-na-do', 
'rostov-na-donu':'rostov-on-do', 
'ribinsk':'rybinsk', 
'ramenskoye':'mo', 
'r.p.isheevka':'isheyevka', 
'pys':'verkhnyaya', 
'prokopevsk':'prokopyevsk', 
'p-kamchatski':'petropavlovs', 
'piatigorsk':'pyatigorsk', 
'pgt':'kuragino', 
'petropavlovsk-k':'petropavlovs', 
'petropavlovsk':'petropavlovs', 
'petropavlovk':'petropavlovs', 
'pervomayskoe':'pervomayskoye', 
'otradnyy':'otradnyj', 
'orel':'oryel', 
'novotroick':'novotroitsk', 
'novokuybyshe':'novokujbyshe', 
'novocheboksa':'novocheboxar', 
'novoaltajsk':'novoaltaysk', 
'novgorod':'n.novgorod', 
'novgoro':'n.novgorod',  
'novg':'n.novgorod', 
'nov.adygeya':'adygeya', 
'nizhn.novgor':'n.novgorod', 
'nizh.novgoro':'n.novgorod', 
'nerungri':'neryungri', 
'nazyvayevsk':'nazyvaevsk', 
'nahodka':'nakhodka', 
'naberezhnye':'nab.chelny', 
'moskva':'moscow', 
'moskva':'moscow', 
'moscow':'moskva', 
'mo':'ramenskoe', 
'mikhaylovsk':'mihajlovsk', 
'maloyaroslav':'maloyaroslavets', 
'lyudinovo':'liudinovo', 
'lukhovitsy':'luhovicy', 
'livny':'livni', 
'lipeck':'lipetsk', 
'likino-dulevo':'likino-dulev', 
'leninsk-kuznets':'leninsk-kuzn', 
'kyakhta':'kjahta', 
'kyahta':'kjahta', 
'kupavna':'kupa', 
'krasnokamensk':'krasnokamens', 
'krasnojarsk':'krasnoyarsk', 
'kopejsk':'kopeysk', 
'komsomolsk-na-a':'komsomolsk-n', 
'komsomolsk':'komsomolsk-n', 
'klintsi':'klintsy', 
'kizil':'kyzyl', 
'kiseliovsk':'kiselevsk', 
'kireevsk':'kireyevsk', 
'kintsy':'klintsy', 
'kholmsk':'holmsk', 
'khimki':'himki', 
'kaspijsk':'kaspiysk', 
'kamensk-shah':'kamensk', 
'kame':'kamen', 
'juzhno-sahal':'y-sakhalinsk', 
'joshkar-ola':'yoshkar-ola', 
'jasnogorsk':'yasnogorsk', 
'jakutsk':'yakutsk', 
'h-sankt-peterburg':'s.-petersbur', 
'habarovsk':'khabarovsk', 
'guryevsk':'gurievsk', 
'elistra':'elista', 
'elec':'elets', 
'ekaterinosla':'yekaterinoslavk', 
'egorevsk':'egorievsk', 
'dolgoprudnyj':'dolgoprudnyy', 
'dolgoprudniy':'dolgoprudnyy', 
'chelyabinsk':'cheliabinsk', 
'chelny':'nab.chelny', 
'cheljabinsk':'cheliabinsk', 
'chelaybinsk':'cheliabinsk', 
'chehov':'chekhov', 
'cheboxary':'cheboksary', 
'ch':'nab.chelny', 
'bronnitsy':'bronnicy', 
'brjansk':'bryansk', 
'blagoveshens':'blagoveshche', 
'blagoveshchensk':'blagoveshche', 
'blagoveschns':'blagoveshche', 
'blagovecshen':'blagoveshche', 
'birobijan':'birobidzhan', 
'bijsk':'biysk', 
'belokuriha':'belokurikha', 
'balashikha':'balashiha', 
'axay':'aksay', 
'archangelsk':'arkhangelsk', 
'almetyevsk':'almetevsk', 
'alexandrov':'aleksandrov', 
'aginskoe':'aginskoye', 
'49/6saratov':'saratov', 
'43/sankt-peterburg':'s.-petersbur', 
'42kaliningrad':'kaliningrad', 
'31neryungri':'neryungri', 
'1volgograd':'volgograd', 
'11sankt-peterburg':'s.-petersbur'
}



def rus_city(s):
#    s='улица А.О. Емельянова, 34, Южно-Сахалинск, Сахалинская область, Россия'
#    s='улица Максима Горького, 20, Тамбов, 111100, Россия'
    l = s.split(',')
    p=[]
    for j in range(len(l)):
        if len(l[j])>1:
            p.append(l[j])
    pp=str()
    for j in range(len(p)-3):       
        pp+=str(p[j]) + ', '
    i=0
    qq=''
    while qq=='':
        i+=1
        if i<len(p):
            q=p[-i].lower()
        
            if 'россия' in q:
                qq=''
            elif 'область' in q:
                qq=''
            elif 'республика' in q:
                qq=''
            elif 'область' in q:
                qq=''
            elif 'край' in q:
                qq=''
            elif 'округ' in q:
                qq=''
            elif 'район' in q:
                qq=''
            elif len(q)<3:
                qq=''
            elif bool(re.search(r'\d', q)):
                qq='' 
            else:
                qq=q
        else: qq="unknown"

    
    return pp.lower().strip(), qq.lower().strip()


X[['street', 'city']]=X['address'].apply(lambda x: x if pd.isnull(x) else a_city(x)).apply(pd.Series)

X['city']=X['city'].apply(lambda x: di[x] if x in di.keys() else x)


X[['street_rus', 'city_rus']]= X['address_rus'].apply(lambda x: x if pd.isnull(x) else rus_city(x)).apply(pd.Series)
dr=['поселок городского типа ',
    'посёлок городского типа ',
    'городское поселение '
    'муниципальное образование город ',
    'муниципальное образование ',
    'особая экономическая зона ',
    ' городское поселение',
    'рабочий поселок ',
    'рабочий посёлок ',
    'сельское поселение ',
    ' сельское поселение',
    'дачный поселок ',
    'дачный посёлок ',
    'город ',
    'село ',
    'деревня ',
    'поселок ',
    'посёлок ',
    'станица ',
    ' кожуун',
    ]
for i in dr:
    X['city_rus'].replace(i,'',inplace=True, regex=True)
    
X['city_rus']=X['city_rus'].str.strip()



b=X['city_rus'].value_counts()

def obl(s):
#    s='улица А.О. Емельянова, 34, Южно-Сахалинск, Сахалинская область, Россия'
#    s='улица Максима Горького, 20, Тамбов, 111100, Россия'
    l = s.split(',')
    p=[]
    for j in range(len(l)):
        if len(l[j])>1:
            p.append(l[j])
    pp=str()
    
    
    for j in range(len(p)-3):       
        pp+=str(p[j]) + ', '
    i=0
    qq=''
    while qq=='':
        i+=1
        if i<len(p):
            q=p[-i].lower()
        
            if 'россия' in q:
                qq=''
            elif bool(re.search(r'\d', q)):
                qq='' 
            elif 'область' in q:
                qq=q
            elif 'республика' in q:
                qq=q
            elif 'область' in q:
                qq=q
            elif 'край' in q:
                qq=q
            elif 'округ' in q:
                qq=''
            elif 'район' in q:
                qq=''
            else:
                qq=q
        else: qq="unknown"
       
    return qq.lower().strip()

X['obl_rus']= X['address_rus'].apply(lambda x: x if pd.isnull(x) else obl(x)).apply(pd.Series)

b=X['obl_rus'].value_counts()


b=X[X['city_rus']=='unknown']



c=list(b['city'].unique())



#
#b.plot()



















#a=X[['city','city_rus']].drop_duplicates()
#b=X[X['city_rus']=='домодедово'][['address_rus', 'address','lat','long']]
#


''' Заполнить адреса которые пропущены'''



df_nan = pd.read_csv('../data/atm_nan.csv', sep=';', encoding='cp866')
df_nan.head()


X = X.set_index('address')
df_nan=df_nan.set_index('address')

X['lat'] = X['lat'].fillna(df_nan['lat'])
X['long'] = X['long'].fillna(df_nan['long'])
X['city_rus'] = X['city_rus'].fillna(df_nan['city_rus'])


X['address']=X.index
#X=X.reset_index(drop=True)
X.index=x_ind

x=[i for i in df_nan['city_rus'].unique() if i not in X['city_rus'].unique()]



a=X[X['long'].isna()]







#
#X['lat']=X.apply(lambda x: df_nan[df_nan['address']==x['address']]['lat'] if x['lat'].isna() else x['lat'] )
#
#X['lat']=np.where(X['lat'].isna(), df_nan[df_nan['address']==X['address']]['lat'], X['lat'])

#a=pd.DataFrame(X.groupby('city_rus')['lat'].mean())
#b=pd.DataFrame(X.groupby('city_rus')['long'].mean())
#c=a.join(b)
#c['city']=c.index



''' unknown 149 '''
X['city_rus']=np.where(X['city_rus']=='unknown', X['city'],X['city_rus'])



'''?????????"""'''

X=X[~((X['lat'].isna()) & (X['isTrain']))]

col='city_rus'
#c=X.groupby(col)['lat'].mean()
#X['lat']=pd.to_numeric(X['lat'])
X['lat'] = X['lat'].fillna(X.groupby(col)['lat'].transform('mean'))
X['long'] = X['long'].fillna(X.groupby(col)['long'].transform('mean'))





c=X[X['lat'].isna()][['address_rus', 'address','lat','long', 'isTrain', 'city', 'city_rus']]
c=c[c['isTrain']==False]['address'].value_counts()



#c=X[X['address'].isin(b.index)]

'''??? TArget encoding '''
#c=df_train.groupby('atm_group')['target'].mean()
#c=pd.DataFrame(c)
#c.columns=['tar']
#c['atm_group']=c.index
#c=c.reset_index(drop=True)

#X=merge_l(X,c,'atm_group')

c=df_train.groupby('atm_group')['target'].mean()
c=pd.DataFrame(c).to_dict()
X['tar']=X['atm_group'].apply(lambda x: c[x] if x in c.keys() else 0)




'''city'''












#_____________________________________________

#_____________________________________________

#_____________________________________________

#_____________________________________________

X['lat'] = X['lat'].fillna(X['lat'].mean())
X['long'] = X['long'].fillna(X['long'].mean())












#
#
#a=pd.DataFrame(X['city'].unique())
#a=list(X['city'].unique())
#a=pd.DataFrame(a)
#a.to_csv('city.csv')

#a=a.apply(lambda x: x.replace(' G ', ''))
#a = df_train[~df_train['address'].isnull()]['address'].apply(lambda x: x[:x.find(' ')])
#
#
#
#
#a=df_train[~df_train['address'].isnull()]['address'].apply(lambda x: pd.Series(x.split('\t')))
#





#Num of ATM's with the same  address
c='address_rus'
counts = X.groupby(c)['id'].count().reset_index().rename(columns={'id':'count'})
counts.columns=[c,'count_addr']
X = pd.merge(X, counts, how='left', on=c)
X['count_addr'].fillna(0, inplace=True)

c='city_rus'
counts = X.groupby(c)['id'].count().reset_index().rename(columns={'id':'count'}).fillna(0)
counts.columns=[c,'count_city']
X = pd.merge(X, counts, how='left', on=c)
X['count_city'].fillna(0, inplace=True)


''' add same for atms in OSM'''



'''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
#Working with target !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
X_g=X[X['isTrain']==True].groupby(['atm_group', 'lat', 'long'])['target'].mean()
X_g=pd.DataFrame(X_g)
X_g.columns=['new_target']
X_g['dr']=X_g.index
X_g[['atm_group', 'lat', 'long']]=X_g['dr'].apply(pd.Series)
X_g.drop(['dr'], axis=1,inplace=True)
X_g.reset_index(drop=True, inplace=True)

X=pd.merge(X, X_g, how='left', on=['atm_group', 'lat', 'long'])
X.drop(['target'], axis=1,inplace=True)
X.rename(columns = {'new_target':'target'}, inplace = True)




'''DROP DUPLICATES'''
'''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''

#x=X[X['isTrain']==True].drop_duplicates(subset=['atm_group', 'lat', 'long'], keep='first')
#X=X[X['isTrain']==False]
#X=pd.concat([x,X], axis=0)

'''ADD NOISE'''
from scipy.stats import truncnorm
X['lat']=X.apply(lambda x: x['lat'] +truncnorm(a=-0.0003, b=0.0003, scale=1).rvs(size=1)[0] if x['count_addr']>1 else x['lat'], axis=1)
X['long']=X.apply(lambda x: x['long'] +truncnorm(a=-0.0003, b=0.0003, scale=1).rvs(size=1)[0] if x['count_addr']>1 else x['long'], axis=1)


#df=X[X['count_addr']>1][['count_addr','lat_c','lat']]


''' -----------------'''


with open('../data/pickle/X.pickle', 'wb') as fout:
   pickle.dump(X, fout, protocol=pickle.HIGHEST_PROTOCOL)

with open('../data/pickle/X.pickle', 'rb') as fin:
    X=pickle.load(fin)
    

'''Tune'''
c='city_rus'
n=5
rare = X[c].value_counts()
rare = X[c].value_counts()[(X[c].value_counts() < n) ==True].index
X[c] = X[c].apply(lambda x: 'RARE' if x in rare else x).fillna('RARE')
X[c+'_rank']= X[c].rank().fillna(-1)


#
#

'''Tune'''
c='obl_rus'
n=8
rare = X[c].value_counts()
rare = X[c].value_counts()[(X[c].value_counts() < n) ==True].index
X[c] = X[c].apply(lambda x: 'RARE' if x in rare else x).fillna('RARE')
X[c+'_rank']= X[c].rank().fillna(-1)



    
    
    #------------------------------------------------





with open('../data/pickle/full_tagged_nodes.pickle', 'rb') as fin:
    tagged_nodes=pickle.load(fin)
    
def unpack(df, column, fillna=None):
    ret = None
    if fillna is None:
        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems()))], axis=1)
        del ret[column]
    else:
        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems())).fillna(fillna)], axis=1)
        del ret[column]
    return ret

#df=unpack(df, 'tags', 0)   


#ATM
raw=0
if raw==1:
    POINT_FEATURE=[('atm', lambda   node:  ( node.tags.get('amenity')=='atm'))]
    
    #'place'== 'village'
    #'place'== 'hamlet'
    #'place'== 'city'
    #'place'== 'town'
    #
    #'name:ru'
    for prefix, point_filter in POINT_FEATURE:
        
        print(prefix)
    #     берем подмножество точек в соответствии с фильтром
        coords = np.array([
            [node.lat, node.lon, 
             #node.tags, 
             node.tags
             
             #node.tags.get('addr:'),
             ]
            for node in tagged_nodes
            if point_filter(node)
        ])
            
    atm=pd.DataFrame(coords)
    
#    , columns=['lat' , 'lon', 'city_rus', 'city', 'population'])
#    atm=atm[~atm['city_rus'].isna()]


    

    
    
    # Сохраним список с выбранными объектами в отдельный файл
    with open('../data/pickle/atm.pickle', 'wb') as fout:
       pickle.dump(atm, fout, protocol=pickle.HIGHEST_PROTOCOL)


#load
with open('../data/pickle/atm.pickle', 'rb') as fin:
    atm=pickle.load(fin)
#name, operator, currency,'cash_in': 'yes', 'currency:RUB': 'yes',
#№банки в крыму которые не существуют, ухтабанк, петрокоммерц
#нет оператора, нет name
raw=0
if raw==1:
    atm=unpack(atm, 2, fillna=None)  
    cols=[0,1,'operator', 'name', 'branch', 'cash_in', 'opening_hours']  
    
    atm=atm[cols]
    atm.columns=['lat','lon','operator', 'name', 'cash_in', 'opening_hours'] 
    atm['operator']=np.where(atm['operator'].isna(),atm['name'], atm['operator'])   
    atm.drop(['name'], axis=1, inplace=True)
    atm['operator']=atm['operator'].str.lower()
    atm['operator']=atm['operator'].str.strip()
    
    #atm['cash_in'].fillna(0, inplace=True)
    atm['cash_in']=np.where(atm['cash_in']=='yes',1, 0)   
    
    
    di={'российский национальный коммерческий банк':'рнкб',
    'банк «россия»':'россия', 
    'аб россия':'россия', 
    'банк россия':'россия',
    'аб россия (ограниченный доступ)':'россия',
    'россия (ограниченный доступ)':'россия',
    
    'оао «ак барс» банк':'ак барс',
    'оао ак барс банк':'ак барс',
    'оао «ак барс»':'ак барс',
    '«ак барс»':'ак барс',
    
    'втб 24':'втб24',
    'втб 24 ()':'втб24',
    'vtb24':'втб24',
    
    'банк втб (пао)':'втб',
    
    
    'оао сбербанк россии':'сбербанк',
    'оао "сбербанк россии"':'сбербанк',
    'сбербанк (ограниченный доступ)':'сбербанк',
    'сбербанк 24часа':'сбербанк',
    'пао "сбербанк"':'сбербанк',
    
    'сбербанк россии':'сбербанк',
    'сбербанк (3 этаж)':'сбербанк',
    'оао Альфа-Банк':'альфа-банк',
    'акб росевробанк (ао)':'росевробанк',
    
    'о альфа-банк':'альфа-банк',
    'rosbank':'росбанк'}
    atm['operator']=atm['operator'].apply(lambda x: di[x] if x in di.keys() else x)
    #atm.columns=['lat','lon','operator', 'cash_in', 'opening_hours'] 
    dr=[
    '\(',
    '\)',
    ','
    '\"',
    '\«',
    '\»',
    'терминал ',
    'банкомат ',
    ' банкомат',
    'пао ',
    'оао ',
    ' пао',
    ' оао',
    'ао ',
    ' ао',
    'банк ',
    ' банк',
    ' рф',
    'фк ',
    'акб ',
    'кб ',
    ' акб',
    ' кб'
    ]
    for i in dr:
        atm['operator'].replace(i,'',inplace=True, regex=True)
    
    
    
    atm['operator'].replace('-',' ',inplace=True, regex=True)
    atm['operator']=atm['operator'].apply(lambda x: di[x] if x in di.keys() else x)
    a_u=atm['operator'].value_counts()
    
    
    
    for i in dr:
        atm['operator'].replace(i,'',inplace=True, regex=True)
    atm['operator']=atm['operator'].str.strip() 
    atm['operator']=atm['operator'].apply(lambda x: di[x] if x in di.keys() else x)
    
    atm['operator'].replace('"','',inplace=True, regex=True)
    
    di={
    'акбарсбанк':'ак барс',
    'акбарс':'ак барс',
    
    'альфабанк':'альфа',
    'alfa bank':'альфа',
    'alfabank':'альфа',
    'алфа':'альфа',
    'альфа, газпромбанк, райффайзен, сбербанк, уралсиб':'альфа',
    'альфа, 24 часа':'альфа',
    'альфа_банк':'альфа',
    'альфа, trust':'альфа',
    
    'vtb':'втб',
    'сбербанк, втб':'втб',
    'втб   москвы':'втб',
    'втб москвы':'втб',
    'москвы':'втб',
    
    'vtb 24':'втб24',
    
    'газпромбанк; двб':'газпромбанк',
    'гаспромбанк':'газпромбанк',
    'газпробанк':'газпромбанк',
    '• газпромбанк':'газпромбанк',
    'газпромкуб':'газпромбанк',
    'газаромбанк':'газпромбанк',
    'газпром':'газпромбанк',
    'газпром/ траст /дальневосточный':'газпромбанк',
    'gazprombank':'газпромбанк',
    'газпромбанк акционерное общество':'газпромбанк',
    'ооо газпромбанк':'газпромбанк',
    'гпб':'газпромбанк',
    'гаспром':'газпромбанк',
    
    'сбер':'сбербанк',
    'сбер24':'сбербанк',
    'сбер24ч':'сбербанк',
    'сбератм+ипк':'сбербанк',
    'сберроссии':'сбербанк',
    'сбербанк,':'сбербанк',
    'сбербанк, 24 часа':'сбербанк',
    'сбербанка':'сбербанк',
    'сберрбанк':'сбербанк',
    'сбербанка россии':'сбербанк',
    'сберроссии':'сбербанк',
    'сберросси':'сбербанк',
    'сберроссии,':'сбербанк',
    'сберрф':'сбербанк',
    'сберюанк':'сбербанк',
    'сербанк':'сбербанк',
    '2а сбербанка россии':'сбербанк',
    'sberbank':'сбербанк',
    'sperbank':'сбербанк',
    'бпс сбербанк':'сбербанк',
    'cбербанк':'сбербанк',
    'отделение сбербанка россии.':'сбербанк',
    'сбербанкомат':'сбербанк',
    'сбербанк;кировский':'сбербанк',
    'сбербанк, мтс':'сбербанк',
    'с.ербанк':'сбербанк',
    'уральский сбербанка россии':'сбербанк',
    
    'россбанк':'росбанк',
    'росбанка':'росбанк',
    
    'россельхозприем выдача':'россельхозбанк',
    'сельхозбанк':'россельхозбанк',
    'росселхозбанк':'россельхозбанк',
    'россельбанк':'россельхозбанк',
    'россельхоз':'россельхозбанк',
    'россельхозбанк':'россельхозбанк',
    '24 россельхозбанк':'россельхозбанк',
    'russian agricultural bank':'россельхозбанк',
    
    'зрайффайзенбанк':'райффайзенбанк',
    'райфазен':'райффайзенбанк',
    'raiffeisen bank aval':'райффайзенбанк',
    'raiffaisenbank':'райффайзенбанк',
    'райффайзен аваль':'райффайзенбанк',
    'райффайзенatm':'райффайзенбанк',
    'raiffaisen':'райффайзенбанк',
    'raiffeisen':'райффайзенбанк',
    'raiffeisen bank':'райффайзенбанк',
    'raiffeisenbank':'райффайзенбанк',
    'райфайзен':'райффайзенбанк',
    'райффазен':'райффайзенбанк',
    'райфайзенбанк':'райффайзенбанк',
    'рабфайзен':'райффайзенбанк',
    'райффайзен':'райффайзенбанк',
    'райффйзен':'райффайзенбанк',
    'раффайзенбанк':'райффайзенбанк',
    
    'уралсиб,':'уралсиб',
    'уралсиб 24часа':'уралсиб',
    'улалсиб':'уралсиб',
    'уралсиб, 24 часа':'уралсиб',
    'уралсиб, 24часа':'уралсиб',
    'уралсиб, мдм':'уралсиб',
    'уралсиббанк':'уралсиб'
    }
    
    atm['operator']=atm['operator'].apply(lambda x: di[x] if x in di.keys() else x)
    
    
    a_u=atm['operator'].value_counts()
    
    
    with open('../data/pickle/atm_opera.pickle', 'wb') as fout:
        pickle.dump(atm, fout, protocol=pickle.HIGHEST_PROTOCOL)


#load
with open('../data/pickle/atm_opera.pickle', 'rb') as fin:
    atm=pickle.load(fin)

atm['lat']=atm['lat'].astype(np.float16)
atm['lon']=atm['lon'].astype(np.float16)

atm_c=atm[~(atm['operator'].isna())] 

atm_sber=atm_c[atm_c['operator']=='сбербанк']

atm_cashin=atm[atm['cash_in']==1]

#atm_sber['coo']=atm.apply(lambda x: str((x['lat'], x['lon'])))
#coo=
#X_s=X[['lat', 'long']]
#X_s.columns=['lat', 'lon']
#X_s['oper']=X_s.apply(lambda x: 1 if (x['lat'],x['lon']).isin(coo) else 0
#
#X_s=merge_l(X_s, atm_sber, ['lat','lon'])
#


#distance to closest SBER


#X_sber=pd.DataFrame(atm_sber[['lat','lon']])
#
#knc = KNeighborsClassifier(metric=distance)
#knc.fit(X=X_sber , y=np.ones(X_sber.shape[0]))
#
#'''tune n_neighbors=6'''
#n=5
#distances, indexes = knc.kneighbors(X=X_sber,n_neighbors=n,)
#for i in range(1,n):
#    X_sber['distance_sber_%s'%i] = distances[:,i]
#    '''????????????????'''
#    X_sber['indexes_sber_%s'%i] = indexes[:,i]
#    '''tune - closest index in same city???'''
#    
#X_sber['mean'] = X_sber.iloc[:,X_sber.columns.str.contains('distance')].mean(axis=1)
#X_sber.drop(['lat', 'long'], axis=1, inplace=True)
#
#
#X = pd.concat([X, X_sber], axis=1)


#----------------------------------------------



def near(X_c, coords,  prefix):
    X_s=pd.DataFrame(X_centers)
    # строим структуру данных для быстрого поиска точек
    neighbors = NearestNeighbors(metric=distance).fit(coords)
    
    # признак вида "количество точек в радиусе R от центра квадрата"
    for radius in [0.1, 0.5, 1, 2]:
        dists, inds = neighbors.radius_neighbors(X=X_centers, radius=radius)
        X_s['{}_points_in_{}'.format(prefix, radius)] = np.array([len(x) for x in inds])
    
    #     признак вида "расстояние до ближайших K точек"
    for n_neighbors in [3, 5, 7, 10]:
    #    for n_neighbors in [1]:
        dists, inds = neighbors.kneighbors(X=X_centers, n_neighbors=n_neighbors)
        if n_neighbors>1:
            X_s['{}_max_dist_k_{}'.format(prefix,n_neighbors)] = dists.max(axis=1)
            X_s['{}_mean_dist_k_{}'.format(prefix, n_neighbors)] = dists.mean(axis=1)
            X_s['{}_std_dist_k_{}'.format(prefix, n_neighbors)] = dists.std(axis=1)
    
    #     признак вида "расстояние до ближайшей точки"
    X_s['{}_min'.format(prefix)] = dists.min(axis=1)
    X_s.drop([0,1], axis=1, inplace=True)
    return X_s


X_centers=X[['lat','long']].as_matrix()
X_centers=X[['lat','long']].values
coords = np.array(atm_sber[['lat','lon']])


X_sber2=near(X_centers, coords,  'sber')
x=[i for i in X_sber2.columns if 'index' in i]
c_sber=list(X_sber2.columns)

#all atms
coords = np.array(atm_c[['lat','lon']])
X_atm=near(X_centers, coords,  'atm_c')


coords = np.array(atm_cashin[['lat','lon']])
X_atm_in=near(X_centers, coords,  'atm_in')

#partner
part=[
'альфа',
'ак барс',
'втб',
'втб24',
'москвы',
'россельхозбанк',
'уралсиб',
'газпромбанк',
'райффайзенбанк',
'росбанк'
]

atm_part=atm[atm['operator'].isin(part)]
a_u=atm_part['operator'].value_counts()
coords = np.array(atm_part[['lat','lon']])
X_atm_part=near(X_centers, coords,  'atm_p')
#nearest within dataset
#coords = np.array(X[['lat','long']])
#X_near=near(X_centers, coords,  'self')
'''-----------------------------------------'''
#___________________________________________



'''-----------------------------------------'''
#___________________________________________

#X = pd.concat([X, X_sber2], axis=1)

#X = pd.concat([X, X_atm], axis=1)

#X = pd.concat([X, X_near], axis=1)

'''-----------------------------------------'''
#___________________________________________






a=df_train['atm_group'].value_counts()
b=df_test['atm_group'].value_counts()










#_____________________________

#Population
raw=1
if raw==0:
    POINT_FEATURE=[('place', lambda   node:  ( node.tags.get('population')))]
    
    #'place'== 'village'
    #'place'== 'hamlet'
    #'place'== 'city'
    #'place'== 'town'
    #
    #'name:ru'
    for prefix, point_filter in POINT_FEATURE:
        
        print(prefix)
    #     берем подмножество точек в соответствии с фильтром
        coords = np.array([
            [node.lat, node.lon, 
             #node.tags, 
             node.tags.get('name:ru'),
             node.tags.get('name:en'),
             node.tags.get('population'), 
     
             #node.tags.get('addr:'),
             ]
            for node in tagged_nodes
            if point_filter(node)
        ])
            
    cities=pd.DataFrame(coords, columns=['lat' , 'lon', 'city_rus', 'city', 'population'])
    cities=cities[~cities['city_rus'].isna()]
    cities=cities[~(cities['population']=='отсутствует')]
    cities=cities[~(cities['population']=='нежилая')]
    cities=cities[~(cities['population']=='100-200')]
    
    cities['city_rus']=cities['city_rus'].str.lower()
    
    
    cities['population']=pd.to_numeric(cities['population'])
    cities['log_pop']=np.log1p(cities['population'])
    
    cities=cities.sort_values(by=['log_pop'], ascending=[0])
    #cities.to_feather('../data/cities_pop.feather')
    
    cities['log_pop_round']=cities['log_pop'].round()
    
    #cities['log_pop'].plot.bar()
    
    
    # Сохраним список с выбранными объектами в отдельный файл
    with open('../data/pickle/cities_pop.pickle', 'wb') as fout:
       pickle.dump(cities, fout, protocol=pickle.HIGHEST_PROTOCOL)

#load
with open('../data/pickle/cities_pop.pickle', 'rb') as fin:
    cities=pickle.load(fin)

cities['city_rus']=cities['city_rus'].str.lower()

X['city_rus']=X['city_rus'].str.lower()

dr=['поселок городского типа ',
'посёлок городского типа ',
'городское поселение '
'муниципальное образование город ',
'муниципальное образование ',
'особая экономическая зона ',
' городское поселение',
'рабочий поселок ',
'рабочий посёлок ',
'сельское поселение ',
' сельское поселение',
'дачный поселок ',
'дачный посёлок ',
'город ',
'село ',
'деревня ',
'поселок ',
'посёлок ',
'станица ',
' кожуун',
]
for i in dr:
    X['city_rus'].replace(i,'',inplace=True, regex=True)

X['city_rus']=X['city_rus'].str.strip()

x=[i for i in X['city_rus'].unique() if i not in cities['city_rus'].unique()]

x=cities[cities['city_rus'].isin(X['city_rus'].unique())][['city_rus','log_pop_round']]




di=cities.set_index('city_rus').to_dict()['log_pop_round']




X['log_pop_round']=X['city_rus'].apply(lambda x: di[x] if x in di.keys() else 0) #map(di).fillna(0)

X['log_pop_round_rank']=X['log_pop_round'].rank().fillna(-1)

X['log_pop_round_per_atm']=X['log_pop_round']/X['count_city']
# add to X population_log/atm_count
    
# add to X population_log/atm_count_OSM   
    
    
    
    
    
    
#BINS , LOG
















#Distance to the closest



#import geopy.distance
#
##coords_1 = (52.2296756, 21.0122287)
##coords_2 = (52.406374, 16.9251681)
##
##print (geopy.distance.geodesic(coords_1, coords_2).km)
#
#def distance(x,y):
#    return geopy.distance.geodesic(x, y).km

#distance(coords_1, coords_2)

knc = KNeighborsClassifier(metric=distance)

dots = X[['lat','long']].dropna()

knc.fit(X=dots , y=np.ones(dots.shape[0]))

#Distatnces between points in dataset
'''tune n_neighbors=6'''
n=7
distances, indexes = knc.kneighbors(X=dots,n_neighbors=n,)

for i in range(1,n):
    dots['distance_%s'%i] = distances[:,i]
    
    dots['distance_%s_'%i] = distances[:,i]
    
    '''????????????????'''
    dots['indexes_%s'%i] = indexes[:,i]
    '''tune - closest index in same city???'''
 

#X_ind=pd.DataFrame(indexes[:,0])
#for i in range(0,7):
#    X_ind[str(i)+'_gr']=X_ind[i].apply(lambda x: X.loc[x]['atm_group'] )
#
#x=[i for i in X_ind.columns if 'gr' in str(i)]
#gr=list(X['atm_group'].unique())
#
#for i in gr:
#    X_ind[str(i)+'_gr_count']=X_ind[x].isin([i]).sum(1)
#    
#X_ind['not_my_type']=np.where(X_ind[0]==X_ind.index, 0,1)  
# 
##X_ind['not_my_1type']=np.where(X_ind[1]==X_ind.index, 0,1) 
#
#X_ind=X_ind.iloc[:,14:]

   
dots['mean'] = dots.iloc[:,dots.columns.str.contains('distance')].mean(axis=1)

dots.drop(['lat', 'long'], axis=1, inplace=True)
#X = pd.concat([X, dots], axis=1)
















#CITY

#rare city


#rare_cities = X.city.value_counts()[(X.city.value_counts() < 20) ==True].index
#
#
#
#X.city = X.city.apply(lambda x: 'RARE' if x in rare_cities else x)
#X.city= X.city.rank().fillna(-1)
#
#









#_________________________________________________________



raw=0
if raw==0:  
    POINT_FEATURE_FILTERS = [
        ('tagged', lambda  node: len(node.tags) > 0),
        ('railway', lambda node: ((node.tags.get('railway') == 'station') |
                                 (node.tags.get('railway') == 'halt')  ) &
                                 (node.tags.get('station') != 'subway')),
        
        ('subway', lambda node: (node.tags.get('railway') == 'station') &
                                 (node.tags.get('station') == 'subway')),
        
        ('public_transport', lambda node: 'public_transport' in node.tags),
        
    
        
        ('pharmacy' , lambda node: (node.tags.get('shop')=='drugstore') |
                                    (node.tags.get('amenity')== 'pharmacy')),
    
        ('rest_bars', lambda node: node.tags.get('amenity') in ['cafe',
                                                                'fast_food',
                                                                'restaurant',
                                                                'bar',
                                                                'pub',
                                                                'nightclub',
                                                                'stripclub',
                                                                'sauna',
                                                                'food_court',
                                                                'love_hotel',
                                                                'club',
                                                                'cafeteria',
                                                                'internet_cafe',
                                                                'biergarten'
                                                                ]),
    
    
        ('amenity_bus_station', lambda node: node.tags.get('amenity')== 'bus_station'),    
        ('amenity_pharmacy', lambda node: node.tags.get('amenity')== 'pharmacy'),
    
       
        ('amenity_bank', lambda node: node.tags.get('amenity')== 'bank'),   
    #    ('amenity_atm', lambda node: node.tags.get('amenity')== 'atm'),
        ('amenity_money_transfer', lambda node: node.tags.get('amenity')== 'money_transfer'),
    
                
        
        
        ('traf_sign', lambda node: node.tags.get('highway')== 'traffic_signals'),
        ('traf_junc', lambda node: node.tags.get('highway')== 'motorway_junction'),
        ('traf_cross', lambda node: (node.tags.get('crossing')== 'uncontrolled') and (node.tags.get('highway')=='crossing')),
    
        ('playground', lambda node: 'playground' in node.tags),
        ('kindergarten', lambda node: (node.tags.get('amenity')=='kindergarten') or (node.tags.get('building')=='kindergarten')),
        ('school', lambda node: (node.tags.get('amenity')=='school') or (node.tags.get('building')=='school')),
        
    #   ('leisure', lambda node: 'leisure' in node.tags),
    
        
        ('office', lambda node: (node.tags.get('building') in ['office','commercial']) |
                                (node.tags.get('amenity') in ['office','business_center'])),
    
        #landuse
    #    ('landuse_industrial', lambda node: node.tags.get('landuse')== 'industrial'),
    #    ('landuse_residential', lambda node: node.tags.get('landuse')== 'residential'),
    #    ('landuse_commercial', lambda node: node.tags.get('landuse')== 'commercial'),
    #    ('landuse_retail', lambda node: node.tags.get('landuse')== 'retail'),
        ('landuse_cemetery', lambda node: node.tags.get('landuse')== 'cemetery'),
    #    ('landuse_farmyard', lambda node: node.tags.get('landuse')== 'farmyard'),
    #    ('landuse_landfill', lambda node: node.tags.get('landuse')== 'landfill'),
    #    ('landuse_military', lambda node: node.tags.get('landuse')== 'military'),
    #    ('landuse_garages', lambda node: node.tags.get('landuse')== 'garages'),
    
        #amenities
        ('amenity_place_of_worship', lambda node: node.tags.get('amenity')== 'place_of_worship'),
        ('amenity_parking', lambda node: node.tags.get('amenity')== 'parking'),
        ('amenity_police', lambda node: node.tags.get('amenity')== 'police'),
        ('amenity_post_office', lambda node: node.tags.get('amenity')== 'post_office'),
        ('amenity_college', lambda node: node.tags.get('amenity')== 'college'),
        ('amenity_university', lambda node: node.tags.get('amenity')== 'university'),
        ('amenity_hospital', lambda node: node.tags.get('amenity')== 'hospital'),
        ('amenity_grave_yard', lambda node: node.tags.get('amenity')== 'grave_yard'),
        ('amenity_office', lambda node: node.tags.get('amenity')== 'office'),
    
    #    ('amenity_bussines_centre', lambda node: node.tags.get('amenity')== 'bussines_centre'),
        ('amenity_marketplace', lambda node: node.tags.get('amenity')== 'marketplace'),
        ('amenity_cafe', lambda node: node.tags.get('amenity')== 'cafe'),
        ('amenity_fast_food', lambda node: node.tags.get('amenity')== 'fast_food'),
        ('amenity_restaurant', lambda node: node.tags.get('amenity')== 'restaurant'),
        ('amenity_bar', lambda node: node.tags.get('amenity')== 'bar'),
        ('amenity_pub', lambda node: node.tags.get('amenity')== 'pub'),
        ('amenity_nightclub', lambda node: node.tags.get('amenity')== 'nightclub'),
        ('amenity_stripclub', lambda node: node.tags.get('amenity')== 'stripclub'),
        ('amenity_sauna', lambda node: node.tags.get('amenity')== 'sauna'),
        ('amenity_food_court', lambda node: node.tags.get('amenity')== 'food_court'),
    
        #shop
        ('shop', lambda node: 'shop' in node.tags),
        ('shop_sup', lambda node: (node.tags.get('shop')== 'supermarket') |
                                    (node.tags.get('shop')== 'convenience') |
                                    (node.tags.get('shop')== 'grocery')),
                
        ('shop_alco', lambda node: node.tags.get('shop') in ['convenience;alcohol',
                                                                'alcohol',
                                                                'beverages',
                                                                'tobacco',
                                                                'wine',
                                                                'beer'
                                                                ]),
        ('shop_mall', lambda node: node.tags.get('shop')== 'mall'),
    
           
        ('finance', lambda node: node.tags.get('amenity') in ['bank', 'atm', 'money_transfer'])
    
    
    ]
    
    x=X.head(100)
    X_centers = X[['lat', 'long']].as_matrix()
    X_centers=X[['lat','long']].values
    
    
    
    
    
    
            
    #_____________________________
    
    
    X_osm=pd.DataFrame(X_centers)
    for prefix, point_filter in POINT_FEATURE_FILTERS:
        
        print(prefix)
    #     берем подмножество точек в соответствии с фильтром
        coords = np.array([
            [node.lat, node.lon]
            for node in tagged_nodes
            if point_filter(node)
        ])
    
        # строим структуру данных для быстрого поиска точек
        neighbors = NearestNeighbors(metric=distance).fit(coords)
        
        # признак вида "количество точек в радиусе R от центра квадрата"
        for radius in [0.1, 0.3, 0.5, 0.7, 1, 2]:
            dists, inds = neighbors.radius_neighbors(X=X_centers, radius=radius)
            X_osm['{}_points_in_{}'.format(prefix, radius)] = np.array([len(x) for x in inds])
    
    #     признак вида "расстояние до ближайших K точек"
        for n_neighbors in [ 3, 5, 7, 10]:
    #    for n_neighbors in [1]:
            dists, inds = neighbors.kneighbors(X=X_centers, n_neighbors=n_neighbors)
            X_osm['{}_mean_dist_k_{}'.format(prefix, n_neighbors)] = dists.mean(axis=1)
            X_osm['{}_max_dist_k_{}'.format(prefix, n_neighbors)] = dists.max(axis=1)
            X_osm['{}_std_dist_k_{}'.format(prefix, n_neighbors)] = dists.std(axis=1)
    
    #     признак вида "расстояние до ближайшей точки"
        X_osm['{}_min'.format(prefix)] = dists.min(axis=1)
    
    
    
    
    X_osm.drop([0,1], axis=1, inplace=True)
    
    x=[i for i in X_osm.columns if 'index' in i]
    c_osm=list(X_osm.columns)
    
    
    with open('../data/pickle/X_osm_dist.pickle', 'wb') as fout:
       pickle.dump(X_osm, fout, protocol=pickle.HIGHEST_PROTOCOL)

#load
with open('../data/pickle/X_osm_dist.pickle', 'rb') as fin:
    X_osm=pickle.load(fin)



dr=['amenity_grave_yard', 'amenity_stripclub', 'amenity_nightclub', 'amenity_sauna', 'landuse_cemetery',
    'playground', 'amenity_money_transfer']

for i in dr:
    dr_c=[x for x in X_osm.columns if i in x]
    X_osm.drop(dr_c, axis=1, inplace=True)



#________________________________________________________


#X.reset_index(drop=True, inplace=True)







from OHE import OHE_single

X_ohe=X[[ 'city_rus', 'obl_rus', 'atm_group', 'log_pop_round']]
cc=[]
for col in  X_ohe.columns:
    print(col)
    X_ohe, c = OHE_single(X_ohe,col,0, True)
    cc.extend(c)


#cols=['atm_group', 'lat', 'long',  'count', 'distance_1',
#                    'distance_2',  'distance_3',  'distance_4', 'distance_5',
#                    'indexes_5', 'mean']



#
#cols=[ 'lat', 'long',   'distance_1',
#                    'distance_2',  'distance_3',  'distance_4',
#                    'distance_5',
#                    'indexes_5', 'mean', 
#                    'count_addr',
#                    'count_city',
#                    'tar',
#                    'log_pop_round_per_atm']


#cols.extend(cc)
#
#X=pd.concat([X,X_osm], axis=1)

#cols.extend(c_osm)

'''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
c=[x for x in dots.columns if 'distance' in x]
dots=dots[c]


#x=df_train[df_train['lat'].isna()]['address']
#xx=df_test[df_test['lat'].isna()]['address']
#x=[i for i in x if i not in xx ]
#
#X=X[~X['address'].isin(x)]
    
#    
#X_f=pd.concat([X,X_atm, dots, X_atm_in, X_atm_part, X_sber2, X_osm, X_ohe], axis=1)


#X_f=pd.concat([X, dots, X_ind,X_atm,  X_atm_in, X_atm_part, X_sber2, X_osm, X_ohe], axis=1)

#no X_ind

#dots.reset_index(drop=True, inplace=True)


X_f=pd.concat([X, dots,X_atm,  X_atm_in, X_atm_part, X_sber2, X_osm, X_ohe], axis=1)

#X_f=X


#X_tr=X_f[X_f.isTrain]
#
#X_tr=X_tr[X_tr['count_addr']>1]
#X_tra=pd.DataFrame(columns=X_f.columns)
#
#X_tr['count_addr'].max()
#
#for i in range(2,int(X_tr['count_addr'].max())+1):
#    df = pd.concat([X_tr[X_tr['count_addr']==i]]*i, ignore_index=True)
#    X_tra = pd.concat([X_tra, df], ignore_index=True)
#
#
#from scipy.stats import truncnorm
#ra= truncnorm(a=-0.0001, b=0.0001, scale=1).rvs(size=1)
#
#X_tra['lat']=X_tra['lat'].apply(lambda x: x +truncnorm(a=-0.0001, b=0.0001, scale=1).rvs(size=1)[0])
#X_tra['long']=X_tra['long'].apply(lambda x: x +truncnorm(a=-0.0001, b=0.0001, scale=1).rvs(size=1)[0])
#
#X_f=pd.concat([X_f[X_f['isTrain']==False], X_tra, X_f[(X_f['isTrain']==True) & (X_f['count_addr']==1)]],ignore_index=True, sort=True, axis=0)
#
#x=list(X_f.columns)
#
#X_tra['lat'].dtype
#df=X_tra[['lat_m', 'lat']]


'''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''

X_f['target']=X_f.apply(lambda x: x['target'] +truncnorm(a=-0.0003, b=0.0003, scale=1).rvs(size=1)[0] if x['count_addr']>1 and x['isTrain']==True else x['target'], axis=1)



Y_ = X_f.loc[X_f.isTrain, 'target']

dr=['address', 'address_rus', 'id', 'isTrain', 'target', 'street_rus', 'street', 'city_rus', 'obl_rus', 'city']
cols=[i for i in X_f.columns if i not in dr]

X_ = X_f[X_f['isTrain']==True][cols]


X_test = X_f[X_f['isTrain']==False][cols]

#X_f.loc[X.isTrain, 'target'].describe()


#X_train, X_valid, Y_train, Y_valid = train_test_split(X_, Y_, test_size=0.25, random_state=1)





x=[i for i in X.columns if i not in X_test.columns]
#x=X[x]


#X_f[X_f.isTrain]['atm_group'].value_counts()

#x=X_f.head()







# Сохраним список с выбранными объектами в отдельный файл
with open('../data/pickle/X_f.pickle', 'wb') as fout:
   pickle.dump(X_f, fout, protocol=pickle.HIGHEST_PROTOCOL)

#load
with open('../data/pickle/X_f.pickle', 'rb') as fin:
    X_f=pickle.load(fin)







#MAjor class by ATM_GROUP

ba=1
if ba==1:
    y_gr= X_f[X_f.isTrain]['atm_group'].astype(str)
if ba==2:
    #X_f['major']=
    y_gr= np.where(X_f[X_f.isTrain].astype(str)=='5478.0',1,0).astype(str)
if ba==3:
    di={'5478.0':1,#    2598
    '1942.0':2,#     1130
    '8083.0':3,#     1031
    '496.5':4,#       613
    '3185.5':4,#      575
    '1022.0':4,#      136
    '32.0':4#         43
    }
    y_gr= X_f[X_f.isTrain]['atm_group'].map(di).astype(str)

from sklearn.model_selection import KFold
see=12134
folds=10
n_e=50000
e_stop=min(n_e*0.05,1000)
l_r=0.001

from sklearn.model_selection import StratifiedKFold

rr={}
from xgboost              import XGBRegressor
import lightgbm as lgb



models=[ 'lgbm',] #, 'xgb']
av_score=0
imp=pd.DataFrame()
imp_x=pd.DataFrame()
scor=pd.DataFrame()
pred=pd.DataFrame(index=df_test.index)

strat=1
l=0
for i in models:



    see=[12134,12645, 776611, 101, 2001]
#    see=[12134,]
    for s in see:
        rr['lgbm'] = lgb.LGBMRegressor(objective = 'regression',  

                            learning_rate = l_r,
                            n_estimators = n_e,
                            metric='rmse',
#                            num_leaves=31,
                            random_state=s,                    
                            max_depth = -1,
                            colsample_bytree = 0.8,
                            subsample = 0.9, 
                            reg_alpha=0.1, 
		                    reg_lambda=0, 

                            nthread=3)





        rr['xgb'] =XGBRegressor(
                        booster='gbtree',
                        base_score=0.5,
        #                min_child_weight:1,
        #                max_leaf_nodes:
                        objective='reg:linear',
                        colsample_bytree = 0.8,
        #                    reg_lambda=1,
        #                    reg_alpha=0.1,
        #                    min_child_weight=2,
        #                    gamma=0.001,
                        learning_rate=l_r,
                        n_estimators=n_e,

#probuem ne fix                   
#                  max_depth=6,
#                  colsample_bytree = 0.8,
                  subsample = 0.9, 


                  metric='rsme',
                  seed=111, 
                  nthread=3,
                  random_state=s) 
        
        
        fol=0
        if strat==1:
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=s)
            sk_f=skf.split(X_, np.array(y_gr))
        else:
            skf = KFold(n_splits=folds, shuffle=True, random_state=s)
            sk_f=skf.split(X_, np.array(Y_))
            
        for train_index, test_index in sk_f:
#        for train_index, test_index in skf.split(X_, Y_):
            fol+=1
            bst=rr[i].fit(X_.iloc[train_index], np.array(Y_.iloc[train_index]),
                        eval_set=[ (X_.iloc[test_index], np.array(Y_.iloc[test_index]))], 
                        eval_metric= 'rmse', 
                        verbose= 500, 
                        early_stopping_rounds= e_stop)
        
        
            if i=='lgbm':
                y_pred = bst.predict(X_.iloc[test_index], num_iteration=bst.best_iteration_)
                '''pred'''
                y_tar=bst.predict(X_test, num_iteration=bst.best_iteration_)
            if i=='xgb':
                y_pred = bst.predict(X_.iloc[test_index], ntree_limit=bst.best_ntree_limit)
                '''pred'''
                y_tar=bst.predict(X_test, ntree_limit=bst.best_ntree_limit)
            
            r=rmse(np.array(Y_.iloc[test_index]), y_pred)
            print('___________________________')
            l+=1
            print(l)
            print(fol, ' ', i, ' RMSE   ', r)
            scor[i+'_'+ '_' +str(fol)+'_'+str(s)]=[r]
            print('___________________________')
            pred[i+'_'+ '_' +str(fol)+'_'+str(s)]=y_tar
            
            av_score+=r
            
            if i=='lgbm':
                imp['l_'+str(fol)+'_'+str(s)]=rr[i].feature_importances_
            if i=='xgb':
                imp_x['x_'+str(fol)+'_'+str(s)]=rr[i].feature_importances_

                        
av_score/=folds*len(see)*len(models)

print('___________________________')
print('RMSE  av ', av_score)







#lgb.plot_importance(gbm)
#
#

imp_x['var']=list(X_test.columns)
imp_x['mean']=imp_x.mean(axis=1)

imp['var']=list(X_test.columns)
imp['mean']=imp_x.mean(axis=1)

#submit = pd.DataFrame(gbm.predict(X_test), index=df_test.index,columns=['target'])


#-----------------------Classification----------------
#from sklearn.metrics import roc_auc_score
#
#from sklearn.linear_model import LogisticRegression
#t=time.clock()        
#classifier = LogisticRegression()
#classifier.fit(X_train, y_train)
#
#import xgboost as xgb
#classifier=xgb.XGBClassifier()
#classifier.fit(X_train, y_train)
#
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 100)
#classifier.fit(X_train, y_train)
#
#from sklearn.svm import SVC
#classifier= SVC(kernel='rbf',random_state=0)
#classifier.fit(X_train,y_train)
#
#from sklearn.ensemble import ExtraTreesClassifier
#classifier = ExtraTreesClassifier(n_estimators = 150, criterion = 'entropy', random_state = 100)
#classifier.fit(X_train, y_train)
#
##------------------regression-----------------------
#
#
from sklearn.linear_model import Ridge
params={'alpha':[0.1,0.001, 0.0005], 'fit_intercept':[False, True]}
model=Ridge()

r=rmse(np.array(Y_.iloc[test_index]), y_pred)
#
#from sklearn.linear_model import BayesianRidge
#params={'alpha_1':[0.001, 0.0005], 'fit_intercept':[False, True]}
#model=BayesianRidge()
#
#
#
#
#
#
#




#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
#td=datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

subm=pd.DataFrame()

#select columns with value gt 0.52
#cols=list(roc.loc[:,(roc>0.56).any()].columns)
#
#cols=[x for x in cols if 'lgbm' in x]
#subm['fin']=pred[cols].mean(axis=1)

xg=[i for i in pred.columns if 'xgb' in i]
lg=[i for i in pred.columns if 'lgbm' in i]

print('xgb')
for row in scor[xg].itertuples(index=False):
    row=list(row)   
    print('MIN: ', min(row))
    print('MAX: ', max(row))
    print('STD: ', np.std(row, axis=0))
    print('MEAN: ', np.mean(row, axis=0))
print (scor[xg].mean(axis=1)[0])
print('lgbm')
for row in scor[lg].itertuples(index=False):
    row=list(row)   
    print('MIN: ', min(row))
    print('MAX: ', max(row))
    print('STD: ', np.std(row, axis=0))
    print('MEAN: ', np.mean(row, axis=0))
  
    

print ('AV xgb: {} , AV lgbm: {}' .format(scor[xg].mean(axis=1)[0], scor[lg].mean(axis=1)[0]) )
i=6
p=0.01 #% to remove from each side
if i==1:
    subm['target']=pred.mean(axis=1)
    cv=av_score
elif i==2:
    subm['target']=pred[xg].mean(axis=1)
    cv=scor[xg].mean(axis=1)[0]
elif i==3:
    subm['target']=pred[lg].mean(axis=1) 
    cv=scor[lg].mean(axis=1)[0]
elif i==4:
      #remove outliers from row
    rez1=[]
    lim=max(int(pred[xg].shape[1]*p),1)
    for row in pred[xg].itertuples(index=False):
        r=sorted(list(row))
        rs=r[lim:len(r)-lim] #trim 1 from each side
        rm=np.mean(rs)
        rez1.append(rm)
    subm['target']=rez1  
    cv=scor[xg].mean(axis=1)[0]
elif i==5:
      #remove outliers from row
    rez1=[]
    lim=max(int(pred[lg].shape[1]*p),1)
    for row in pred[lg].itertuples(index=False):
        r=sorted(list(row))
        rs=r[lim:len(r)-lim] #trim 1 from each side
        rm=np.mean(rs)
        rez1.append(rm)
    subm['target']=rez1 
    cv=scor[lg].mean(axis=1)[0]
    
elif i==6:
    #remove outliers from row
    cols=list(scor.loc[:,(scor<0.0415).any()].columns)
    subm['target']=pred[cols].mean(axis=1)
    
    
else:
      #remove outliers from row
    rez1=[]
    lim=max(int(pred.shape[1]*p),1)
    for row in pred.itertuples(index=False):
        r=sorted(list(row))
        rs=r[lim:len(r)-lim] #trim 1 from each side
        rm=np.mean(rs)
        rez1.append(rm)
    subm['target']=rez1  
    cv=av_score


import datetime
td=datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
f = open('myfile.txt', 'a')
st='{} Shape 1: {} fol {} strat {} AV_score {} i {} cv {} | {} {} {} {} {} {} {} {}'.format(td, X_test.shape[1],
    folds, strat, av_score, i, cv, dots.shape[1], X_atm.shape[1], 
    X_ind.shape[1],  X_atm_in.shape[1], X_atm_part.shape[1], X_sber2.shape[1],
    X_osm.shape[1], X_ohe.shape[1])
f.write(st+'\n')
f.close()

   
#imp_x.to_csv(os.path.dirname(os.getcwd())+'\\subm\\___imp_x_s_'+str(strat)+'CV_'+ str(int(round(cv*1000000,0)))+ '_i_'+str(i)+'_bal_'+str(ba)+ '_'+ td+'.csv', index=True, header=True)
imp.to_csv(os.path.dirname(os.getcwd())+'\\subm\\____imp_l_s_'+str(strat)+'CV_'+ str(int(round(cv*1000000,0)))+ '_i_'+str(i)+'_bal_'+str(ba)+ '_'+ td+'.csv', index=True, header=True)

pred.to_csv(os.path.dirname(os.getcwd())+'\\subm\\___pred_s_'+str(strat)+'CV_'+ str(int(round(cv*1000000,0)))+ '_i_'+str(i)+'_bal_'+str(ba)+ '_'+ td+'.csv', index=True, header=True)
  
scor.to_csv(os.path.dirname(os.getcwd())+'\\subm\\___score_s_'+str(strat)+'CV_'+ str(int(round(cv*1000000,0)))+ '_i_'+str(i)+'_bal_'+str(ba)+ '_'+ td+'.csv', index=True, header=True)
subm.to_csv(os.path.dirname(os.getcwd())+'\\subm\\subm_s_'+str(strat)+'CV_'+ str(int(round(cv*1000000,0)))+ '_i_'+str(i)+'_bal_'+str(ba)+ '_'+ td+'.csv', index=True, header=True)


from save_zip import save_src_to_zip

save_src_to_zip(os.path.dirname(os.getcwd())+'\\src_zip\\',  exclude_folders = ['__pycache__'], dname="src", td=td)




