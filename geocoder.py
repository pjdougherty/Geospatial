# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:04:36 2015

@author: pdougherty
"""

import pandas as pd
from pygeocoder import Geocoder as geo
import time


def promptFilePath():
    path = raw_input('Please make sure the CSV being read contains headers named "Address," "Latitude," and "Longitude."\nPaste filepath: ')
    if path[-3:]!='csv' and path[-3:]!='CSV':
        print 'Please paste only paths to CSV files'
        return
    else:
        pass
    estab_df = pd.read_csv('%s' % path, header=0) # write csv filepath to load data
    return estab_df

def geocode():
    input_type = raw_input('Are you hoping to geocode addresses or coordinates?\nEnter Addresses or Coordinates: ')
    results = []
    estab_df = promptFilePath()
    if input_type=='Addresses':
        print 'Fetching coordinates...'
        for a in estab_df["Address"]:
            time.sleep(0.25)
            results.append(list(geo.geocode(a).coordinates))
        geocoded_estab_df = estab_df.join(pd.DataFrame(results, columns=['Latitude', 'Longitude']))
    elif input_type=='Coordinates':
        print 'Fetching addresses...'
        for i,c in estab_df.iterrows():
            time.sleep(0.25)
            results.append(list(geo.reverse_geocode(estab_df.Latitude[i], estab_df.Longitude[i])))
        geocoded_estab_df = estab_df.join(pd.DataFrame(results, columns=['Address']))
    else:
        print 'Please only select Addresses or Coordinates.'
        return
    return geocoded_estab_df
    
def saveGeocodedFile():
    geocoded_estab_df = geocode()
    save_filepath = raw_input('Paste CSV filepath to save geocoded data as a new document: ')
    if save_filepath[-3:]!='csv' and save_filepath[-3:]!='CSV':
        print 'I can only save your geocoded data as a CSV. Please only paste a filepath to a new CSV.'
        return
    else:
        pass
    try:
        geocoded_estab_df.to_csv('%s' % save_filepath)
    except:
        print 'There\'s an issue with saving your file. Make sure you\'re attempting to save as a new file.'
        return
        
saveGeocodedFile()