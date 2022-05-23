# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 18:50:42 2018

@author: Yury
"""


from sklearn.neighbors import NearestNeighbors

import pandas as pd
import numpy as np
import pickle


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
    ('amenity_bank', lambda node: node.tags.get('amenity')== 'bank'),   
    ('amenity_atm', lambda node: node.tags.get('amenity')== 'atm'),
    ('amenity_pharmacy', lambda node: node.tags.get('amenity')== 'pharmacy'),
    ('amenity_money_transfer', lambda node: node.tags.get('amenity')== 'money_transfer'),

            
    
    
    ('traf_sign', lambda node: node.tags.get('highway')== 'traffic_signals'),
    ('traf_junc', lambda node: node.tags.get('highway')== 'motorway_junction'),
    ('traf_cross', lambda node: (node.tags.get('crossing')== 'uncontrolled') and (node.tags.get('highway')=='crossing')),

    ('playground', lambda node: 'playground' in node.tags),
    ('kindergarten', lambda node: (node.tags.get('amenity')=='kindergarten') or (node.tags.get('building')=='kindergarten')),
    ('school', lambda node: (node.tags.get('amenity')=='school') or (node.tags.get('building')=='school')),
    
    ('leisure', lambda node: 'leisure' in node.tags),

    
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

    ('amenity_bussines_centre', lambda node: node.tags.get('amenity')== 'bussines_centre'),
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
    ('shop_mall', lambda node: node.tags.get('shop')== 'mall')
]


l=[
   'crimean-fed-district-latest.osm.pbf',
   'far-eastern-fed-district-latest.osm.pbf',
   'northwestern-fed-district-latest.osm.pbf',
'central-fed-district-latest.osm.pbf',

'ural-fed-district-latest.osm.pbf',
'volga-fed-district-latest.osm.pbf',
'south-fed-district-latest.osm.pbf',
'siberian-fed-district-latest.osm.pbf',
'north-caucasus-fed-district-latest.osm.pbf'


]


     
import time       
import osmread
from tqdm import tqdm_notebook

#LAT_MIN, LAT_MAX = min(X['lat'].min()), max(X['lat'].max())
#LON_MIN, LON_MAX = min(X['long'].min()), max(X['long'].max())

#print( LAT_MIN, LAT_MAX , LON_MIN, LON_MAX)
raw=1
if raw==0:
    for i in l:
        print (i)
        print (time.clock())
        osm_file = osmread.parse_file('../data/'+i)
                    
        tagged_nodes = [
            entry
            for entry in tqdm_notebook(osm_file, total=18976998)
            if isinstance(entry, osmread.Node)
            if len(entry.tags) > 0
        #    if (LAT_MIN < entry.lat < LAT_MAX) and (LON_MIN < entry.lon < LON_MAX)
        ]
        
        '''nodes'''
    
        
        # Сохраним список с выбранными объектами в отдельный файл
        with open('../data/pickle/'+i[:-8]+'_tagged_nodes.pickle', 'wb') as fout:
            pickle.dump(tagged_nodes, fout, protocol=pickle.HIGHEST_PROTOCOL)


tagged_nodes_full=[]
for i in l: 
#    i=l[1]
    print(i) 
    with open('../data/pickle/'+i[:-8]+'_tagged_nodes.pickle', 'rb') as fin:
        tagged_nodes_full.extend(pickle.load(fin)) 

tmp=tagged_nodes_full[:10000]

coords={}
for prefix, point_filter in POINT_FEATURE_FILTERS:

#     берем подмножество точек в соответствии с фильтром
    coords[prefix] = np.array([
        [node.lat, node.lon]
        for node in tagged_nodes_full
        if point_filter(node)
    ])
            
am=[]
for i in list(coords.keys()):
    if len(coords[i])>0:
        pd.DataFrame(coords[i],columns=['lat', 'lon']).to_csv('../data/'+i+'.csv', index=False)
        am.append(i)






            
# Сохраним список с выбранными объектами в отдельный файл
with open('../data/pickle/full_tagged_nodes.pickle', 'wb') as fout:
   pickle.dump(tagged_nodes_full, fout, protocol=pickle.HIGHEST_PROTOCOL)

            

temp=[] 

def unp_df(ll, ddf):
    for i in ll: 
#        i=l[1]
        print(i) 
        with open('../data/pickle/'+i[:-8]+'_tagged_nodes.pickle', 'rb') as fin:
            
    #        tagged_nodes_full.extend(pickle.load(fin)) 
            tmp=unpack(pd.DataFrame(pickle.load(fin)), 'tags', 0)

            
            ddf=pd.concat([ddf,tmp], axis=0)
    return df
            
x_keep=['id', 'lon', 'lat',
'addr:district',
'addr:region',
'admin_level',
#'alt_name',
'name',
#'nat_name',
#'official_name',
'opening_hours',
'place',
'population',
'public_transport',
'railway',
'shop',
'traffic_signals',
'train'
]


def unpack(df, column, fillna=None):
    ret = None
    if fillna is None:
        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems()))], axis=1)
        del ret[column]
    else:
        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[column].iteritems())).fillna(fillna)], axis=1)
        del ret[column]
        
#    x_drop=[x for x in ret.columns if ('name:' in x) or ('is_in' in x) or x in ['source', 'highway']]   
#    ret.drop(x_drop, axis=1, inplace=True)
#    ret=ret[x_keep]
    ret['id'] = ret['id'].astype(np.int8)
    for col in ['lat', 'lon']:
        ret[col] = ret[col].astype(np.float16)
    return ret

df=pd.DataFrame(columns=x_keep)


df_shop=pd.DataFrame(columns=x_keep)

# Файл с сохраненными объектами можно будет быстро загрузить




df=unp_df(l[:2], df)
df.to_feather('../data/tags.feather')



df=pd.from_feather('../data/tags.feather')
for col in ['lat', 'lon']:
    df[col] = df[col].astype(np.float16)
df['id'] = df['id'].astype(np.int8)

#for i in range(0,10):
#    temp = pd.DataFrame(tagged_nodes_full[i*500000:(i+1)*500000])
#
#    




##open
#with open('../data/pickle/full_tagged_nodes.pickle', 'rb') as fin:
#   tagged_nodes_full=pickle.load(fin) 

len(tagged_nodes_full)
temp=tagged_nodes_full[:100]






#df = pd.DataFrame(temp)



tagged_nodes_full=[]

#c=list(df.columns)

#a=list(df.columns)
#

X_centers = X[['lat_c', 'lon_c']].as_matrix()

X_osm=pd.DataFrame(X_centers)
for prefix, point_filter in POINT_FEATURE_FILTERS:

#     берем подмножество точек в соответствии с фильтром
    coords = np.array([
        [node.lat, node.lon]
        for node in tagged_nodes
        if point_filter(node)
    ])

    # строим структуру данных для быстрого поиска точек
    neighbors = NearestNeighbors().fit(coords)
    
    # признак вида "количество точек в радиусе R от центра квадрата"
    for radius in [0.001, 0.003, 0.005, 0.007, 0.01]:
        dists, inds = neighbors.radius_neighbors(X=X_centers, radius=radius)
        X_osm['{}_points_in_{}'.format(prefix, radius)] = np.array([len(x) for x in inds])

#     признак вида "расстояние до ближайших K точек"
    for n_neighbors in [3, 5, 10]:
#    for n_neighbors in [1]:
        dists, inds = neighbors.kneighbors(X=X_centers, n_neighbors=n_neighbors)
        X_osm['{}_mean_dist_k_{}'.format(prefix, n_neighbors)] = dists.mean(axis=1)
        X_osm['{}_max_dist_k_{}'.format(prefix, n_neighbors)] = dists.max(axis=1)
        X_osm['{}_std_dist_k_{}'.format(prefix, n_neighbors)] = dists.std(axis=1)

#     признак вида "расстояние до ближайшей точки"
    X_osm['{}_min'.format(prefix)] = dists.min(axis=1)
