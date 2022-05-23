# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:33:50 2018

@author: Yury
"""

#from imposm.parser import OSMParser
#
#p.parse('../data/RU.osm.pbf')
#
#
import ogr
#import shapely
#from shapely.geometry import *
#import geopandas as gpd
#import matplotlib.pyplot as plt
#http://andrewgaidus.com/Convert_OSM_Data/
#

driver=ogr.GetDriverByName('OSM')
data = driver.Open('inner_richmond.osm')
layer = data.GetLayer('points')

features=[x for x in layer]
print ( len(features))


data_list=[]
for feature in features:
    data=feature.ExportToJson(as_object=True)
    coords=data['geometry']['coordinates']
    shapely_geo=Point(coords[0],coords[1])
    name=data['properties']['name']
    highway=data['properties']['highway']
    other_tags=data['properties']['other_tags']
    if other_tags and 'amenity' in other_tags:
        feat=[x for x in other_tags.split(',') if 'amenity' in x][0]
        amenity=feat[feat.rfind('>')+2:feat.rfind('"')]
    else:
        amenity=None
        
    data_list.append([name,highway,amenity,shapely_geo])
gdf=gpd.GeoDataFrame(data_list,columns=['Name','Highway','Amenity','geometry'],crs={'init': 'epsg:4326'}).to_crs(epsg=3310)


gdf.tail()

#cafe_bar=gdf[gdf.Amenity.isin(['cafe','pub','bar'])]
#cafe_bar
#
#
#fig, ax = plt.subplots(figsize=(10,10))
#for i,row in cafe_bar.iterrows():
#    x=row['geometry'].x
#    y=row['geometry'].y
#    plt.annotate(row['Name'], xy=(x,y), size=13, xytext=(0,5), textcoords='offset points')
#    plt.plot(x,y,'o', color='#f16824')
#    ax.set(aspect=1)
#title=plt.title('Inner Richmond Bars and Cafes',size=16)

from osmread import parse_file, Node

l=[]
c=0
for entity in parse_file('../data/RU.osm.pbf'):
    while c<10:
        if isinstance(entity, Node) and 'amenity' in entity.tags:
            l.append(entity)
            c+=1
        