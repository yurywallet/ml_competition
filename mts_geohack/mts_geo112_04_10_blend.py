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


from sklearn.model_selection import cross_val_score

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



df_zones = pd.read_csv('../data/zones.csv', index_col='zone_id', engine="python")

y_target=df_zones.iloc[:,11:]

'''read points with tags from paticular region'''
import osmread
from tqdm import tqdm_notebook
import pickle

min_lat=min(df_zones['lat_bl'].min(),df_zones['lat_tr'].min())
max_lat=max(df_zones['lat_bl'].max(),df_zones['lat_tr'].max())

min_lon=min(df_zones['lon_bl'].min(),df_zones['lon_tr'].min())
max_lon=max(df_zones['lon_bl'].max(),df_zones['lon_tr'].max())
print( min_lat, max_lat)

LAT_MIN, LAT_MAX = min_lat, max_lat
LON_MIN, LON_MAX = min_lon, max_lon

#LAT_MIN, LAT_MAX = 55.309397, 56.13526
#LON_MIN, LON_MAX = 36.770379, 38.19270

#osm_file = osmread.parse_file('../data/RU-MOS.osm.pbf')
#tagged_nodes = [
#    entry
#    for entry in tqdm_notebook(osm_file, total=18976998)
#    if isinstance(entry, osmread.Node)
#    if len(entry.tags) > 0
#    if (LAT_MIN < entry.lat < LAT_MAX) and (LON_MIN < entry.lon < LON_MAX)
#]
#
#'''nodes'''
#
#with open('../data/tagged_nodes.pickle', 'wb') as fout:
#    pickle.dump(tagged_nodes, fout, protocol=pickle.HIGHEST_PROTOCOL)
#    

#tagged_way = [
#    entry
#    for entry in tqdm(osm_file, total=18976998)
#    if isinstance(entry, osmread.Way)
#]
#with open('../data/tagged_way.pickle', 'wb') as fout:
#    pickle.dump(tagged_way, fout, protocol=pickle.HIGHEST_PROTOCOL)

with open('../data/tagged_nodes.pickle', 'rb') as f:
         tagged_nodes = pickle.load(f)
#df = pd.DataFrame(tagged_nodes)
#
#def unpack(df, column, fillna=None):
#    ret = None
#    if fillna is None:
#        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems()))], axis=1)
#        del ret[column]
#    else:
#        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems())).fillna(fillna)], axis=1)
#        del ret[column]
#    return ret
#
#df=unpack(df, 'tags', 0)


df = pd.read_csv('../data/nodes_streetmap.csv',  engine="python")

df=memory_reduce(df)


'''count_tags'''
osm_tags = []
for i in range(len(tagged_nodes)):
    if (len(tagged_nodes[i].tags)>0):
        for j in tagged_nodes[i].tags:
            osm_tags.append(j)
tags, count = np.unique(osm_tags,return_counts=True)

tags_count = pd.DataFrame({'tag':tags, 
                           'count':count}).sort_values(by='count',ascending=False).reset_index(drop=True)

#####################################################3   
a=list(df.columns)

df['building'].unique()
df['building:flats'].unique()
c='amenity'
df[c].unique()
c='highway'
df[c].unique()
c='junction'

c='place'
b=df[c].value_counts()



'''---count objects in zone-----'''

def zone_map(df1, df2):
    df2['zone_id'] = df2.index
    for col in ['zone_id', 'lat_bl', 'lon_bl', 'lat_tr', 'lon_tr']:
        df1[col]=np.zeros(df1.shape[0])
        
    for t ,t2 in df2.iterrows():
        mask=(df1['lat'] >=df2.loc[t,'lat_bl'])  & (df1['lat'] <df2.loc[t,'lat_tr']) & (df1['lon'] >=df2.loc[t,'lon_bl']) & (df1['lon'] <df2.loc[t,'lon_tr'])
        for col in ['zone_id', 'lat_bl', 'lon_bl', 'lat_tr', 'lon_tr']:
            df1.loc[mask, col] = df2.loc[t,col]    
    return df1

df=zone_map(df, df_zones)
df.to_csv('../data/nodes_streetmap_zones.csv')


#df=pd.read_csv('../data/nodes_streetmap_zones.csv')
df=df[df['zone_id']!=0]

df[df['zone_id']!=0]['zone_id'].value_counts().sum()

#df=df[df['power']!= 'tower']
#df=df[df['highway']!= 'traffic_signals']
#
#
#
#a=len(fd)
#print (fd[0].get('shop')=='supermarket')
#df=df[df['tags'].str.contains("'power': 'tower'")==False]


#if izb in zone then sum 'size'

'''таблицу с простыми признаками, которые можно использовать для предсказания'''
import collections 

df_features = collections.OrderedDict([])

'''distance to Kremlin'''
import math

kremlin_lat, kremlin_lon = 55.753722, 37.620657
moscow_lat, moscow_lon = 55.7522200, 37.6155600

#https://www.movable-type.co.uk/scripts/latlong.html
def dist_calc(lat1, lon1, lat2, lon2):
    R = 6373.0 #metres

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

df_features['distance_to_kremlin'] = df_zones.apply(
    lambda row: dist_calc(row.lat_c, row.lon_c, kremlin_lat, kremlin_lon), axis=1)

city_lat, city_lon = 55.7411, 37.5918
df_features['distance_to_city'] = df_zones.apply(
    lambda row: dist_calc(row.lat_c, row.lon_c, city_lat, city_lon), axis=1)


df_features['distance_to_kremlin'].plot()


#df1=df_features.head()


#mayakovsk_lon, mayakovsk_lat =37.5951107,	55.7702196
#dist_calc(mayakovsk_lat, mayakovsk_lon, kremlin_lat, kremlin_lon)
#mkad 19
#sadovoe 3
#ttk 5
#df_features['sadovoe']=np.where(df_features['distance_to_kremlin']<3,1,0)
df_features['bulv']=(df_features['distance_to_kremlin']<1)*1
df_features['bulv'].value_counts()
df_features['sadovoe']=((df_features['distance_to_kremlin']>=1) & (df_features['distance_to_kremlin']<2.5))*1
df_features['sadovoe'].value_counts()
df_features['ttk']=((df_features['distance_to_kremlin']>=2.5) & (df_features['distance_to_kremlin']<5))*1
df_features['ttk'].value_counts()
df_features['mkad']=((df_features['distance_to_kremlin']>=5) & (df_features['distance_to_kremlin']<19))*1
df_features['mkad'].value_counts()

df_features['zamkad30']=((df_features['distance_to_kremlin']>=19) & (df_features['distance_to_kremlin']<30))*1
df_features['zamkad30'].value_counts()
df_features['zamkad30plus']=((df_features['distance_to_kremlin']>=30) )*1
df_features['zamkad30plus'].value_counts()

from sklearn.neighbors import NearestNeighbors


# набор фильтров точек, по которым будет считаться статистика
POINT_FEATURE_FILTERS = [
    ('tagged', lambda  node: len(node.tags) > 0),
    ('railway', lambda node: ((node.tags.get('railway') == 'station') or
                             (node.tags.get('railway') == 'halt')  ) and
                             (node.tags.get('station') != 'subway')),
    
    ('subway', lambda node: (node.tags.get('railway') == 'station') &
                             (node.tags.get('station') == 'subway')),
     
    ('bus_stop', lambda node: node.tags.get('highway')== 'bus_stop'),
    
    ('public_transport', lambda node: 'public_transport' in node.tags),
    
#    ('shop', lambda node: 'shop' in node.tags),
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
    
    ('pharmacy' , lambda node: (node.tags.get('shop')=='drugstore') or
                                (node.tags.get('amenity')== 'pharmacy')),
    
    ('traf_sign', lambda node: node.tags.get('highway')== 'traffic_signals'),
    ('traf_junc', lambda node: node.tags.get('highway')== 'motorway_junction'),
    ('traf_cross', lambda node: ((node.tags.get('crossing')== 'uncontrolled') or  
                                     (node.tags.get('crossing')== 'unmarked'))
    and   (node.tags.get('highway')=='crossing') ),
    
    ('traf_cam', lambda node: node.tags.get('highway')== 'speed_camera'),

#    ('playground', lambda node: 'playground' in node.tags),
#    ('kindergarten', lambda node: (node.tags.get('amenity')=='kindergarten') or (node.tags.get('building')=='kindergarten')),
#    ('school', lambda node: (node.tags.get('amenity')=='school') or (node.tags.get('building')=='school')),
#    
    ('leisure', lambda node: 'leisure' in node.tags),
    #added cuisine
    ('rest_bars', lambda node: (node.tags.get('amenity') in ['cafe',
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
                                                            ]) or ('cuisine' in node.tags)),
    ('healthcare', lambda node: (node.tags.get('amenity') in ['hospital','clinic','dentist','doctors']) or
                                ('healthcare' in node.tags) ),

    
    ('education', lambda node: node.tags.get('amenity') in ['kindergarten', 'school','college','university','education'
                                , 'supplementary_education','musical_school', 'additional_education','education_centre']
                                ),
    ('parking', lambda node: node.tags.get('amenity') in['parking','parking_space','parking_entrance','motorcycle_parking']),

    
    ('tourism', lambda node: 'tourism' in node.tags),
    
    ('emergency', lambda node: ('fire_operator' in node.tags) 
                                or (node.tags.get('emergency')== 'fire_hydrant') 
                                or (node.tags.get('amenity')== 'fire_station:type=spo')
                                or (node.tags.get('amenity')== 'fire_station')
                                or (node.tags.get('amenity')== 'police')
                                ),
    ('bank_atm', lambda node: ('atm' in node.tags) or (node.tags.get('amenity') in ['bank', 'atm'])
    ),
    ('name', lambda node: 'name' in node.tags),
    ('place', lambda node: 'place' in node.tags),
    
    ('public_transport_stop_position', lambda node: node.tags.get('public_transport') == 'stop_position')

            
]
    


# центры квадратов в виде матрицы
X_zone_centers = df_zones[['lat_c', 'lon_c']].as_matrix()

for prefix, point_filter in POINT_FEATURE_FILTERS:
    print(prefix)
    # берем подмножество точек в соответствии с фильтром
    coords = np.array([
        [node.lat, node.lon]
        for node in tagged_nodes
        if point_filter(node)
    ])

    # строим структуру данных для быстрого поиска точек
    neighbors = NearestNeighbors().fit(coords)
    
    # признак вида "количество точек в радиусе R от центра квадрата"
    for radius in [0.001, 0.003, 0.005, 0.007, 0.01]:
        dists, inds = neighbors.radius_neighbors(X=X_zone_centers, radius=radius)
        df_features['{}_points_in_{}'.format(prefix, radius)] = np.array([len(x) for x in inds])

    # признак вида "расстояние до ближайших K точек"
    for n_neighbors in [3, 5, 10]:
        '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
#    for n_neighbors in [1]:
        dists, inds = neighbors.kneighbors(X=X_zone_centers, n_neighbors=n_neighbors)
        df_features['{}_mean_dist_k_{}'.format(prefix, n_neighbors)] = dists.mean(axis=1)
        df_features['{}_max_dist_k_{}'.format(prefix, n_neighbors)] = dists.max(axis=1)
        df_features['{}_std_dist_k_{}'.format(prefix, n_neighbors)] = dists.std(axis=1)

    # признак вида "расстояние до ближайшей точки"
    df_features['{}_min'.format(prefix)] = dists.min(axis=1)
    

#
#
#
#
#
#''' less frequent features'''
POINT_FEATURE_FILTERS = [    
    ('office', lambda node: (node.tags.get('building') in ['office','commercial']) |
                            (node.tags.get('amenity') in ['office','business_center'])),

    #
#    ('landuse_industrial', lambda node: node.tags.get('landuse')== 'industrial'),
#    ('landuse_residential', lambda node: node.tags.get('landuse')== 'residential'),
#    ('landuse_commercial', lambda node: node.tags.get('landuse')== 'commercial'),
#    ('landuse_retail', lambda node: node.tags.get('landuse')== 'retail'),
#    ('landuse_cemetery', lambda node: node.tags.get('landuse')== 'cemetery'),
#    ('landuse_farmyard', lambda node: node.tags.get('landuse')== 'farmyard'),
#    ('landuse_landfill', lambda node: node.tags.get('landuse')== 'landfill'),
#    ('landuse_military', lambda node: node.tags.get('landuse')== 'military'),
#    ('landuse_garages', lambda node: node.tags.get('landuse')== 'garages'),

    #amenities
    ('amenity_place_of_worship', lambda node: node.tags.get('amenity')== 'place_of_worship'),

#    ('amenity_police', lambda node: node.tags.get('amenity')== 'police'),
    
    ('amenity_register_office', lambda node: node.tags.get('amenity')== 'register_office'),
    ('amenity_prison', lambda node: node.tags.get('amenity')== 'prison'),
    
    ('amenity_dormitory', lambda node: node.tags.get('amenity')== 'dormitory'),
    ('amenity_restaurant', lambda node: node.tags.get('amenity')== 'restaurant'),
    ('amenity_bar', lambda node: node.tags.get('amenity')== 'bar'),
    ('amenity_pub', lambda node: node.tags.get('amenity')== 'pub'),
    ('amenity_stripclub', lambda node: node.tags.get('amenity')== 'stripclub'),
    
    ('amenity_cinema', lambda node: node.tags.get('amenity')== 'cinema'),
    ('amenity_gym', lambda node: node.tags.get('amenity')== 'gym'),
    ('amenity_beauty', lambda node: node.tags.get('amenity')== 'beauty'),
    
    
    ('amenity_school', lambda node: node.tags.get('amenity')== 'school'),
    ('amenity_college', lambda node: node.tags.get('amenity')== 'college'),
    ('amenity_university', lambda node: node.tags.get('amenity')== 'university'),
    ('amenity_education', lambda node: node.tags.get('amenity')== 'education'),
    ('amenity_supplementary_education', lambda node: node.tags.get('amenity')== 'supplementary_education'),
    ('amenity_musical_school', lambda node: node.tags.get('amenity')== 'musical_school'),
    ('amenity_additional_education', lambda node: node.tags.get('amenity')== 'additional_education'),
    ('amenity_education_centre', lambda node: node.tags.get('amenity')== 'education_centre'),
    ('amenity_parking', lambda node: node.tags.get('amenity')== 'parking'),
    ('amenity_parking_space', lambda node: node.tags.get('amenity')== 'parking_space'),
    ('amenity_parking_entrance', lambda node: node.tags.get('amenity')== 'parking_entrance'),
    ('amenity_motorcycle_parking', lambda node: node.tags.get('amenity')== 'motorcycle_parking'),
    ('amenity_veterinary', lambda node: node.tags.get('amenity')== 'veterinary'),
    ('amenity_pet', lambda node: node.tags.get('amenity')== 'pet'),
    ('amenity_hospital', lambda node: node.tags.get('amenity')== 'hospital'),
    ('amenity_clinic', lambda node: node.tags.get('amenity')== 'clinic'),
    ('amenity_dentist', lambda node: node.tags.get('amenity')== 'dentist'),
    ('amenity_crematorium', lambda node: node.tags.get('amenity')== 'crematorium'),
    ('amenity_mortuary', lambda node: node.tags.get('amenity')== 'mortuary'),
    ('amenity_bank', lambda node: node.tags.get('amenity')== 'bank'),
    ('amenity_atm', lambda node: node.tags.get('amenity')== 'atm'),
    ('amenity_post_office', lambda node: node.tags.get('amenity')== 'post_office'),

    ('crematorium', lambda node: (node.tags.get('amenity')== 'crematorium') or   
                         (node.tags.get('amenity')== 'mortuary')),

    #new add
#    ('street', lambda node: 'addr:housenumber' in node.tags),
#
#    ('housenum', lambda node: 'addr:street' in node.tags)
    
    #    'bank'+'atm'

 
#    'entrance' ['staircase', 'yes', 'main']

]
# центры квадратов в виде матрицы
X_zone_centers = df_zones[['lat_c', 'lon_c']].as_matrix()

for prefix, point_filter in POINT_FEATURE_FILTERS:

    # берем подмножество точек в соответствии с фильтром
    coords = np.array([
        [node.lat, node.lon]
        for node in tagged_nodes
        if point_filter(node)
    ])

    # строим структуру данных для быстрого поиска точек
    neighbors = NearestNeighbors().fit(coords)
    
    # признак вида "количество точек в радиусе R от центра квадрата"
    for radius in [0.001, 0.003, 0.005, 0.007, 0.01]:
        dists, inds = neighbors.radius_neighbors(X=X_zone_centers, radius=radius)
        df_features['{}_points_in_{}'.format(prefix, radius)] = np.array([len(x) for x in inds])

    # признак вида "расстояние до ближайших K точек"
#    for n_neighbors in [3, 5, 10]:
    '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
    for n_neighbors in [1]:
        dists, inds = neighbors.kneighbors(X=X_zone_centers, n_neighbors=n_neighbors)
        df_features['{}_mean_dist_k_{}'.format(prefix, n_neighbors)] = dists.mean(axis=1)
        df_features['{}_max_dist_k_{}'.format(prefix, n_neighbors)] = dists.max(axis=1)
        df_features['{}_std_dist_k_{}'.format(prefix, n_neighbors)] = dists.std(axis=1)

    # признак вида "расстояние до ближайшей точки"
    df_features['{}_min'.format(prefix)] = dists.min(axis=1)
    

'''-------------------------------------------------------------'''

df_features = pd.DataFrame(df_features, index=df_zones.index)
df_features['zone_id']=df_features.index


'''-------------------------------------------------------------'''

'''cams'''
cams = pd.read_csv('../data/camery_clean.csv', sep=';')

def zone_map(df1, df2):
    df2['zone_id'] = df2.index
    for col in ['zone_id', 'lat_bl', 'lon_bl', 'lat_tr', 'lon_tr']:
        df1[col]=np.zeros(df1.shape[0])
        
    for t ,t2 in df2.iterrows():
        mask=(df1['lat'] >=df2.loc[t,'lat_bl'])  & (df1['lat'] <df2.loc[t,'lat_tr']) & (df1['lon'] >=df2.loc[t,'lon_bl']) & (df1['lon'] <df2.loc[t,'lon_tr'])
        for col in ['zone_id', 'lat_bl', 'lon_bl', 'lat_tr', 'lon_tr']:
            df1.loc[mask, col] = df2.loc[t,col]    
    return df1
cams=zone_map(cams, df_zones)
cams.to_csv('../data/cams_zones.csv')
'''home_cams'''
cams_home = pd.read_csv('../data/camery_dom_clean.csv', sep=';')
cams_home=zone_map(cams_home, df_zones)
cams_home.to_csv('../data/cams_home_zones.csv')

'''population'''

#izb = pd.read_csv('../data/data_izbirkom.csv')
#izb = izb[(izb.region == 'moscow_city') | (izb.region == 'moscow_reg')]
#def zone_map_iz(df1, df2):
#    df2['zone_id'] = df2.index
#    for col in ['zone_id', 'lat_bl', 'lon_bl', 'lat_tr', 'lon_tr']:
#        df1[col]=np.zeros(df1.shape[0])
#        
#    for t ,t2 in df2.iterrows():
#        mask=(df1['latitude'] >=df2.loc[t,'lat_bl'])  & (df1['latitude'] <df2.loc[t,'lat_tr']) & (df1['longitude'] >=df2.loc[t,'lon_bl']) & (df1['longitude'] <df2.loc[t,'lon_tr'])
#        for col in ['zone_id', 'lat_bl', 'lon_bl', 'lat_tr', 'lon_tr']:
#            df1.loc[mask, col] = df2.loc[t,col]    
#    return df1
#izb=zone_map_iz(izb, df_zones)
#izb.to_csv('../data/izb_zones.csv')
izb = pd.read_csv('../data/izb_zones.csv')
izb=izb[izb['zone_id']!=0]
#a=izb[izb['zone_id']==8679]['size']

'''population'''
a=pd.DataFrame(izb.groupby('zone_id')['size'].sum())
a['zone_id']=a.index
a=a.rename(columns={'size':'popul'})
df_features=df_features.merge(a, on='zone_id', how='left').fillna(0)
#df_features.drop(['popul'], axis=1, inplace=True)
'''shop'''
#a=pd.DataFrame(df[df['shop']=='supermarket'][['zone_id','shop']])
#a=a.groupby('zone_id').size().reset_index(name='count_super')
#df_features=df_features.merge(a, on='zone_id', how='left').fillna(0)


'''pharma'''
#a=pd.DataFrame(df[(df['shop']=='drugstore') | (df['amenity']=='pharmacy')][['zone_id','shop']])
#a=a.groupby('zone_id').size().reset_index(name='count_pharma')
#df_features=df_features.merge(a, on='zone_id', how='left').fillna(0)

'''cam_home'''
a=pd.DataFrame(cams_home.groupby('zone_id')['global_id'].size().reset_index(name='count_cams_home'))
df_features=df_features.merge(a, on='zone_id', how='left').fillna(0)
#df_features.drop(['count_cams_home'], axis=1, inplace=True)

'''cams'''
a=pd.DataFrame(cams.groupby('zone_id')['global_id'].size().reset_index(name='count_cams'))
df_features=df_features.merge(a, on='zone_id', how='left').fillna(0)


#dtp
a=pd.DataFrame(df_dtp.groupby('zone_id')['reg_code'].size().reset_index(name='count_dtp'))
df_features=df_features.merge(a, on='zone_id', how='left').fillna(0)



#df_features.drop(['count_cams'], axis=1, inplace=True)

''' homes'''
#col='addr:housenumber'
#a=pd.DataFrame(df[(df[col]!=0)][['zone_id',col]])
#a=a.groupby('zone_id').size().reset_index(name='count_homes')
#df_features=df_features.merge(a, on='zone_id', how='left').fillna(0)


#df_features['shop_popul']=np.where(df_features['popul']>0,df_features['shop']/df_features['popul'],0)
'''-------------------------------------------------------------'''
#zones_home = pd.read_csv('../data/raif_home_count.csv', index_col='zone_id', engine="python")
#zones_work = pd.read_csv('../data/raif_work_count.csv', index_col='zone_id', engine="python")
#
##join works with index, merge with columns
#df_features=df_features.join(zones_home,   how='left').fillna(0)
#df_features=df_features.join(zones_work,   how='left').fillna(0)
#
#df_features.drop(['counts_work'], axis=1, inplace=True)
'''-------------------------------------------------------------'''



df_features.to_csv('../data/features.csv')

df_features.head()




#MODEL
df_features=memory_reduce(df_features)

target_columns = ['calls_wd{}'.format(d) for d in range(7)]



df_zones_train = df_zones.query('is_test == 0 & is_target == 1')
#df_zones_train = df_zones.query('is_test == 0 ')

'''-----------------------------------------'''
'''box-cox'''
target_mod=[]
for col in target_columns:
#    A value is trying to be set on a copy of a slice from a DataFrame.
#    df_zones_train.loc[:,col+'_mod']=np.sqrt(np.log1p(df_zones_train[col])).reset_index()
    a=pd.DataFrame(index=df_zones_train.index, columns=[col+'_mod'])
    a [col+'_mod']=np.sqrt(np.log1p(df_zones_train[col]))
    df_zones_train = df_zones_train.join(a,  how='left').fillna(0)
    target_mod.append(col+'_mod')
'''------'''
'''box-cox'''
target_modrank=[]
for col in target_columns:
    a=pd.DataFrame(index=df_zones_train.index, columns=[col+'_modRank'])
    a [col+'_modRank']=df_zones_train[col+'_mod'].rank(ascending=True) #,method='dense')
    df_zones_train = df_zones_train.join(a,  how='left').fillna(0)
    target_modrank.append(col+'_modRank')
'''-----------------------------------------'''
target_rank=[]
for col in target_columns:
    a=pd.DataFrame(index=df_zones_train.index, columns=[col+'_Rank'])
    a [col+'_Rank']=df_zones_train[col].rank(ascending=True) #,method='dense')
    df_zones_train = df_zones_train.join(a,  how='left').fillna(0)
#    
#    df_zones_train.loc[:,col+'_Rank'] = df_zones_train.loc[:,col].rank(ascending=True)
#    df_zones_train.loc[:,col+'_Rank1'] = df_zones_train.loc[:,col].transform(lambda x: x.rank())
    target_rank.append(col+'_Rank')
     

'''-----------------------------------------'''
'''split'''   
from sklearn.model_selection import train_test_split 
idx_train, idx_valid = train_test_split(df_zones_train.index, test_size=0.1, random_state=121)


idx_test = df_zones.query('is_test == 1').index
df_test_predictions = pd.DataFrame(index=idx_test)
df_train_predictions= pd.DataFrame(index=df_features.index)

X_train = df_features.loc[idx_train, :]
X_valid = df_features.loc[idx_valid, :]
X_test = df_features.loc[idx_test, :]


from sklearn.preprocessing import MinMaxScaler
mimasc=MinMaxScaler()
X_train=mimasc.fit_transform(X_train)
X_valid=mimasc.transform(X_valid)
X_test=mimasc.transform(X_test)

X_train_all=mimasc.transform(df_features)

from sklearn.model_selection import GridSearchCV
'''-------------------'''
from sklearn.metrics import make_scorer
from scipy.stats import kendalltau
    
def scorer(y, y_pr):
#    check_estimator(estimator)
#    y_pr=estimator.predict(x)
    return kendalltau(y, y_pr).correlation

score_rank=make_scorer(scorer , greater_is_better=True)

d={}
#first_best
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, n_jobs=4, random_state=121)
#params={'n_estimators':[400,1000],'min_samples_leaf':[1,5,9], 'max_depth':[None,10]}
params={'n_estimators':[500],'min_samples_leaf':[1,5], 'max_depth':[None,10]}

d['forest']=(params, model)

#second best
import xgboost as xgb
#params={'gamma':[0, 0.01], 'max_depth':[9,13,17,21],
##            'booster': ['gblinear', 'dart']
#        'objective':['rank:pairwise', 'rank:ndcg'] #, #, 'multi:softprob'
##        ,'eta':[0.01,0.005]
#        ,'colsample_bytree':[0.9, 1]
#         ,'learning_rate':[0.01,0.05],
#         'n_estimators':[200,400,1000,2000]}

params={'gamma':[0], 'max_depth':[13,17,21],
#            'booster': ['gblinear', 'dart']
        'objective':['rank:pairwise'] #, #, 'multi:softprob'
#        ,'eta':[0.01,0.005]
        ,'colsample_bytree':[1]
         ,'learning_rate':[0.01],
         'n_estimators':[500]}

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, 
                        objective='rank:pairwise', seed=121
                       #, eval_metric=score_rank
                       )
d['xgb']=(params, model)



#from catboost import CatBoostRegressor
#model = CatBoostRegressor(iterations=10,loss_function='MAE')
#params={'depth':[2,3,5]}
#d['cat']=(params, model)


'''target contains daily, weekday, weekend, and w0...w6'''

tar_all=['calls_daily','calls_weekday','calls_weekend', 
         'calls_w0','calls_w1', 'calls_w2', 'calls_w3', 'calls_w4',
          'calls_w5', 'calls_w6']


target_list=[(target_mod, 'mod')
#            (target_modrank, 'modrank'),
#             ,(target_rank, 'rank')
            ]

models_name=['xgb','forest'] #,'cat']
models_list=[ (d['xgb'],'xgb'), (d['forest'], 'forest')] #(d['cat'], 'cat'),



for mod, mod_nam in models_list: 
    model=mod[1]
    params=mod[0]
    print(mod_nam)
    for cols, nam in target_list:
        print(nam)
        for col in cols:
            y_train = df_zones_train.loc[idx_train, col]
            y_valid = df_zones_train.loc[idx_valid, col]
            
            gs = GridSearchCV(estimator=model,
                                          param_grid=params,
                                          scoring=score_rank,
                                          n_jobs=1,
                                          cv=5,
                                          verbose=1)
            gs.fit(X_train, y_train)
            print(gs.best_params_)
            print(gs.best_score_)
            
            #validation
            y_valid = df_zones_train.loc[idx_valid, col]
            y_pred = gs.predict(X_valid)

#            df_valid_target = df_zones_train.loc[idx_valid, col]
#            
#            df_valid_predictions = pd.DataFrame(collections.OrderedDict([(column_name, y_pred)
#                for column_name in cols]), index=idx_valid)
#                
#            df_comparison = pd.DataFrame({
#                'target': df_valid_target.unstack(),
#                'prediction': df_valid_predictions.unstack(),
#            })
            valid_score=kendalltau(y_valid, y_pred).correlation
#            valid_score = kendalltau(df_comparison['target'], df_comparison['prediction']).correlation
            print('Validation score:', valid_score)
        
            #predict test
            y_pred = gs.predict(X_test)
            print(col+'_'+mod_nam+'_'+nam)
            df_test_predictions[col+'_'+mod_nam+'_'+nam]=y_pred
            
            #predict TRAIN 
            y_pred = gs.predict(X_train_all)
            print(col+'_'+mod_nam+'_'+nam)
            df_train_predictions[col+'_'+mod_nam+'_'+nam]=y_pred
    
#####################################################################

df_test_predictions.to_csv(os.path.dirname(os.getcwd())+'\\data\\df_pred_test.csv', index=True)        
'''blending'''
#cols like "calls_daily_modrank"
#mean of two models
#df_test_predictions['calls_wd0_mod_xgb_mod']

pred_blend=pd.DataFrame(index=idx_test)
for cols, nam in target_list:
    for col in cols: 
        col_n=[col+'_'+mod_nam+'_'+nam for mod_nam in models_name]
        pred_blend[col+'_'+nam]=df_test_predictions[col_n].mean(axis=1)

#mean of tree targets
#pred_blend_t=pd.DataFrame(index=idx_test)
#for col in tar_all: 
#    col_n=[col+'_'+nam for nam in target_list]
#    pred_blend_t[col]=pred_blend[col_n].mean(axis=1)  

#mean to final pediction   
#tar_weekday=['calls_daily','calls_weekday',  
#         'calls_w0','calls_w1', 'calls_w2', 'calls_w3', 'calls_w4']
#tar_weekend=['calls_daily', 'calls_weekend','calls_w5', 'calls_w6']
tar_weekday=['calls_w0','calls_w1', 'calls_w2', 'calls_w3', 'calls_w4']
tar_weekend=['calls_w5', 'calls_w6']    
  
pred_blend.columns = ['calls_w0','calls_w1', 'calls_w2', 'calls_w3', 'calls_w4', 'calls_w5', 'calls_w6']
df_test_final=pd.DataFrame(index=idx_test)
for i in range(5):
    df_test_final["calls_wd"+str(i)]=pred_blend[tar_weekday].mean(axis=1)

for i in [5,6]:
    df_test_final["calls_wd"+str(i)]=pred_blend[tar_weekend].mean(axis=1)    

a=0
for col in pred_blend.columns:
    
    a+=1
#---------------------------------------------------------------------------
import datetime
import os
td=datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

df_test_final.to_csv(os.path.dirname(os.getcwd())+'\\subm\\submission_blend_fin'+ '_'+ td+'.csv', index=True)

pred_blend.to_csv(os.path.dirname(os.getcwd())+'\\subm\\submission_blend'+ '_'+ td+'.csv', index=True)

#---- stack



from sklearn.linear_model import Ridge
params={'alpha':[0.1,0.001, 0.0005], 'fit_intercept':[False, True]}
model=Ridge()

from sklearn.linear_model import BayesianRidge
params={'alpha_1':[0.001, 0.0005], 'fit_intercept':[False, True]}
model=BayesianRidge()

df_test_final_stack = pd.DataFrame(index=idx_test)
av_score=0
for col in target_columns:
    #select all columns that contain calls_wd0
    pred=df_train_predictions.filter(regex=col)
    
    gs = GridSearchCV(estimator=model,
                                          param_grid=params,
                                          scoring=score_rank,
                                          n_jobs=1,
                                          cv=10,
                                          verbose=0)
    
    y_train = df_zones_train.loc[idx_train, col]
    y_valid = df_zones_train.loc[idx_valid, col]
    
    X_train = pred.loc[idx_train, :]
    X_valid = pred.loc[idx_valid, :]      
    
    gs.fit(X_train, y_train)
    
    #validation
    y_valid = df_zones_train.loc[idx_valid, col]
    y_pred = gs.predict(X_valid)

#    df_valid_predictions = pd.DataFrame(collections.OrderedDict([(column_name, y_pred)
#        for column_name in cols]), index=idx_valid)
#        
#    df_comparison = pd.DataFrame({
#        'target': y_valid.unstack(),
#        'prediction': df_valid_predictions.unstack(),
#    })
#
#    valid_score = kendalltau(df_comparison['target'], df_comparison['prediction']).correlation
    valid_score=kendalltau(y_valid, y_pred).correlation
    print('Validation score:', valid_score)
    av_score+=valid_score
    #predict test
    X_test=df_test_predictions.filter(regex=col)
    y_pred = gs.predict(X_test)
    df_test_final_stack[col]=y_pred

print('Validation average score:', av_score/7)  

df_test_final_stack.to_csv(os.path.dirname(os.getcwd())+'\\subm\\submission_stack'+ '_'+ td+'.csv', index=True)

#-------
df_test_final_stack.columns = ['calls_w0','calls_w1', 'calls_w2', 'calls_w3', 'calls_w4', 'calls_w5', 'calls_w6']
df_test_final=pd.DataFrame(index=idx_test)
for i in range(5):
    df_test_final["calls_wd"+str(i)]=df_test_final_stack[tar_weekday].mean(axis=1)

for i in [5,6]:
    df_test_final["calls_wd"+str(i)]=df_test_final_stack[tar_weekend].mean(axis=1)    

df_test_final.to_csv(os.path.dirname(os.getcwd())+'\\subm\\submission_stack_mean'+ '_'+ td+'.csv', index=True)



#df_test_predictions["wd_mean"]=df_test_predictions.iloc[:,:5].mean(axis=1)
#df_test_predictions["we_mean"]=df_test_predictions.iloc[:,5:7].mean(axis=1)
#
#df_test_mean=pd.DataFrame(index=idx_test)
#for i in range(5):
#    df_test_mean["calls_wd"+str(i)]=df_test_predictions["wd_mean"]
#
#for i in [5,6]:
#    df_test_mean["calls_wd"+str(i)]=df_test_predictions["we_mean"]
#
#df_test_predictions["wd_median"]=df_test_predictions.iloc[:,:5].median(axis=1)
#df_test_predictions["we_median"]=df_test_predictions.iloc[:,5:7].median(axis=1)



#   
#  
#df_test_median=pd.DataFrame(index=idx_test)
#for i in range(5):
#    df_test_median["calls_wd"+str(i)]=df_test_predictions["wd_mean"]
#
#for i in [5,6]:
#    df_test_median["calls_wd"+str(i)]=df_test_predictions["we_mean"]   
##---------------------------- 
#df_test_predictions["wd_max"]=df_test_predictions.iloc[:,:5].max(axis=1)
#df_test_predictions["we_max"]=df_test_predictions.iloc[:,5:7].max(axis=1)
#df_test_max=pd.DataFrame(index=idx_test)
#
#for i in range(5):
#    df_test_max["calls_wd"+str(i)]=df_test_predictions["wd_max"]
#
#for i in [5,6]:
#    df_test_max["calls_wd"+str(i)]=df_test_predictions["we_max"]    
#    
##--
#import datetime
#td=datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
#
#df_test_predictions.to_csv(os.path.dirname(os.getcwd())+'\\subm\\submission'+ '_'+ td+'.csv', index=True)
#
#df_test_mean.to_csv(os.path.dirname(os.getcwd())+'\\subm\\submission_mean'+ '_'+ td+'.csv', index=True)
#
#df_test_median.to_csv(os.path.dirname(os.getcwd())+'\\subm\\submission_median'+ '_'+ td+'.csv', index=True)
#
#df_test_max.to_csv(os.path.dirname(os.getcwd())+'\\subm\\submission_max'+ '_'+ td+'.csv', index=True)


'''visualisation'''
import folium

fmap = folium.Map([55.753722, 37.620657])

# нанесем ж/д станции
for node in tagged_nodes:
    if node.tags.get('railway') == 'station':
        folium.CircleMarker([node.lat, node.lon], radius=3).add_to(fmap)

# выделим квадраты с наибольшим числом ;жителей
col='popul'
df_map=pd.DataFrame(index=df_zones.index)
df_map['zone_id']=df_map.index
df_map[col]=df_features[col]
for i in ['lat_bl', 'lon_bl', 'lat_tr', 'lon_tr']:
    df_map[i]=df_zones[i]
thresh = df_features.popul.quantile(.50)
for _, row in df_map.query('popul > @thresh').iterrows():
    folium.features.RectangleMarker(
        bounds=((row.lat_bl, row.lon_bl), (row.lat_tr, row.lon_tr)),
        fill_color='red',
    ).add_to(fmap)
#    
col='count_dtp'
b=df_features[col].value_counts()
print(b)
df_map=pd.DataFrame(index=df_zones.index)
df_map['zone_id']=df_map.index
df_map[col]=df_features[col]
for i in ['lat_bl', 'lon_bl', 'lat_tr', 'lon_tr']:
    df_map[i]=df_zones[i]
thresh = 1
for _, row in df_map.query('count_dtp > @thresh').iterrows():
    folium.features.RectangleMarker(
        bounds=((row.lat_bl, row.lon_bl), (row.lat_tr, row.lon_tr)),
        fill_color='yellow',
    ).add_to(fmap)
thresh = 2
for _, row in df_map.query('count_dtp > @thresh').iterrows():
    folium.features.RectangleMarker(
        bounds=((row.lat_bl, row.lon_bl), (row.lat_tr, row.lon_tr)),
        fill_color='pink',
    ).add_to(fmap)
thresh = 3   
for _, row in df_map.query('count_dtp > @thresh').iterrows():
    folium.features.RectangleMarker(
        bounds=((row.lat_bl, row.lon_bl), (row.lat_tr, row.lon_tr)),
        fill_color='red',
    ).add_to(fmap)
    
# выделим квадраты с наибольшим числом вызовов
calls_thresh = df_zones.calls_daily.quantile(.9999)
for _, row in df_zones.query('calls_daily > 60').iterrows():
    folium.features.RectangleMarker(
        bounds=((row.lat_bl, row.lon_bl), (row.lat_tr, row.lon_tr)),
        fill_color='red',
    ).add_to(fmap)


fmap = folium.Map([55.753722, 37.620657])
col='calls_wd0'
df_map=pd.DataFrame(index=df_test_final.index)
df_map['zone_id']=df_map.index
df_map[col]=df_test_final[col]
for i in ['lat_bl', 'lon_bl', 'lat_tr', 'lon_tr']:
    df_map[i]=df_zones[i]

thresh3 = df_test_final[col].quantile(.75)
thresh = df_test_final[col].quantile(.90)
thresh2 = df_test_final[col].quantile(.95)

for _, row in df_map.query('calls_wd0 > @thresh3 & calls_wd0 <= @thresh').iterrows():
    folium.features.RectangleMarker(
        bounds=((row.lat_bl, row.lon_bl), (row.lat_tr, row.lon_tr)),
        fill_color='#ff99cc',
    ).add_to(fmap)
    
for _, row in df_map.query('calls_wd0 > @thresh & calls_wd0 < @thresh2').iterrows():
    folium.features.RectangleMarker(
        bounds=((row.lat_bl, row.lon_bl), (row.lat_tr, row.lon_tr)),
        fill_color='#ff6666',
    ).add_to(fmap)

for _, row in df_map.query('calls_wd0 >= @thresh2').iterrows():
    folium.features.RectangleMarker(
        bounds=((row.lat_bl, row.lon_bl), (row.lat_tr, row.lon_tr)),
        fill_color='#990000',
    ).add_to(fmap)    


# карту можно сохранить и посмотреть в браузере
fmap.save('../maps/map_1'+td+'.html')

fmap

c='calls_weekend'
b=df_zones_train[c].value_counts()

b=df_zones_train[c].describe()

df_zones_train[df_zones_train[c]>70]['zone_id']

#----

df_random = pd.read_csv('../subm/submission_mean_2018-04-14_18_49_25.csv', index_col='zone_id', engine="python")
df_xgb = pd.read_csv('../subm/submission_mean_2018-04-15_00_08_14.csv', index_col='zone_id', engine="python")

df_test=pd.DataFrame(index=idx_test)
for i in range(7):
    df_test["calls_wd"+str(i)]=(df_random["calls_wd"+str(i)]+df_xgb["calls_wd"+str(i)])/2

df_test.to_csv(os.path.dirname(os.getcwd())+'\\subm\\submission_2bl'+ '_'+ td+'.csv', index=True)



from save_zip import save_src_to_zip

save_src_to_zip(os.path.dirname(os.getcwd())+'\\src_zip\\',  exclude_folders = ['__pycache__'], dname="src", td=td)

