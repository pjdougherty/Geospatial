# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:21:01 2015

@author: pdougherty
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:22:07 2015

@author: pdougherty
"""

import pandas as pd
import numpy as np
import scipy.spatial.distance as dist
import math

col_names = pd.read_csv(r'G:\Publications\2015\Regional Clustering\NETS2012_MD_EAGB.txt', nrows=1, delimiter='\t')
col_names = list(col_names.columns)
# Turn this on ONLY when an EAGB industry has been assigned to each company
col_names.append('EAGBIndustry')
move_col_names = 'DunsNumber	MoveYear	Company	TradeName	MoveSIC	OriginCity	OriginState	OriginZIP	OriginFIPSCounty	OriginCounty	DestCity	DestState	DestZIP	DestFIPSCounty	DestCounty	MoveEmp	EmpC	MoveSales	MoveSalesC	MoveOften	Active	SizeCat	EstCat	Subsidiary	OriginLatitude	OriginLongitude	OriginLevelCode	DestLatitude	DestLongitude	DestLevelCode	Distance'
move_col_names = move_col_names.split('\t')
move_col_names = list(move_col_names)
for i in range(0, len(move_col_names)):
    move_col_names[i] = move_col_names[i] + '_r'

all_col_names = col_names+move_col_names

def transformDataFrame(df):
    # Read in and merge dataframe with relocation information
    NETS_move = pd.read_csv(r'G:\Publications\2015\Regional Clustering\NETS Industry Group Industries\nets_move.csv', names=move_col_names)
    df = pd.merge(df, NETS_move, left_on='DunsNumber', right_on='DunsNumber_r', how='left')    
    # Adjust and zip Origin, Destination, and All establishment coordinates
    # First, get negative longitude values because NETS uses all positive longitudes
    df["Longitude"] = 0-df["Longitude"]
    df["OriginLongitude"] = 0-df["OriginLongitude_r"]
    df["DestLongitude"] = 0-df["DestLongitude_r"]
    df['OriginLatitude'] = df['OriginLatitude_r']
    df['DestLatitude'] = df['DestLatitude_r']
    df = df.reset_index(drop=True)
    return df
    
def defineLargeEmployers(df, year):
    sortby = 'Emp'+str(year)[2:4]
    cutoff = df.sort(sortby, ascending=False).head(25).tail(1)[sortby]
    cutoff = int(cutoff)
    df = df[df['%s' % sortby] >= cutoff]
    df = df.reset_index(drop=True)
    return df
            
def getLatLon(df, year='None', landmark='none'):        
    df['lat'] = np.where(df.LastMove.isnull()==True, df.Latitude, df.OriginLatitude)
    df['lon'] = np.where(df.LastMove.isnull()==True, df.Longitude, df.OriginLongitude)
    # Convert degrees to radians
    df['rlat'] = df.lat*math.pi/180
    df['rlon'] = df.lon*math.pi/180
    
    return df
    
def getCartesianCoords(df):
    X = 3959*np.cos(df.rlat)*np.cos(df.rlon)
    y = 3959*np.cos(df.rlat)*np.sin(df.rlon)
    z = 3959*np.sin(df.rlat)
    
    return np.array(zip(X, y, z))


def calcEuclideanDistance(df, lmk_df, lmk, sum_col_name, short_col_name):       
    distances = []
    shortest_distances = []    
    estabs = getCartesianCoords(df)
    

    for i, estab in df.iterrows():
        if np.isnan(df['LastMove'][i]) == False:
            year = df.LastMove[i]
        else:
            year = 2012
            
        if lmk=='Top 25 Employers':
            lmk_df = defineLargeEmployers(df, year)
        else:
            lmk_df = lmk_df[lmk_df.FirstYear<=year][lmk_df.LastYear>=year]
            
        lmk_loc = getCartesianCoords(lmk_df)
        
        total_distance = 0
        short_distance = []
        for lmk in lmk_loc:
            distance = dist.euclidean(estabs[i], lmk)
            short_distance.append(distance)
            total_distance += distance
        distances.append(total_distance)
        shortest_distances.append(sorted(short_distance)[0])
        
    return pd.DataFrame(zip(distances, shortest_distances), columns=[sum_col_name, short_col_name])


'''  The following calculates Euclidean Distance between establishments and all landmarks.
        The one above calculcates the distance between establishments and the landmarks that existed
        when they moved or are open in 2012 if they haven't moved
'''
'''
def calcEuclideanDistance(df, lmk_df, sum_col_name, short_col_name):
    estabs = getCartesianCoords(df)
    lmks = getCartesianCoords(lmk_df)
    
    distances = []
    shortest_distances = []
    for estab in estabs:
        total_distance = 0
        short_distance = []
        for lmk in lmks:
            distance = dist.euclidean(estab, lmk)
            short_distance.append(distance)
            total_distance += distance
        distances.append(total_distance)
        shortest_distances.append(sorted(short_distance)[0])
        
    return pd.DataFrame(zip(distances, shortest_distances), columns=[sum_col_name, short_col_name])
'''
 
def getEuclideanDistance(df, lmk, sum_col_name, short_col_name):
    if lmk=='Hospitals':
        landmark = pd.read_csv(r'G:\Publications\2015\Regional Clustering\NETS Industry Group Industries\hospitals.csv', names=col_names, index_col=False)
    elif lmk=='Universities':
        landmark = pd.read_csv(r'G:\Publications\2015\Regional Clustering\NETS Industry Group Industries\colleges and universities.csv', names=col_names, index_col=False)
    elif lmk=='Top 25 Employers':
        landmark = 'None'
    
    print 'Transforming the establishment dataframe...'
    try:
        estab = transformDataFrame(df)
    except:
        estab = df
    
    print 'Transforming the landmark dataframe...'
    try:
        landmark = transformDataFrame(landmark)
    except:
        pass
    
    print 'Getting latitude and longitude coordinates of establishments and landmarks...\n'
    if lmk != 'Top 25 Employers':
        estab, landmark = getLatLon(estab), getLatLon(landmark)
    else:
        estab = getLatLon(estab)
    
    distances = calcEuclideanDistance(estab, landmark, lmk, sum_col_name, short_col_name)
    
    return estab.join(distances)