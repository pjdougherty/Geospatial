# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:41:37 2015

@author: pdougherty
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
from pysal.esda.mapclassify import Natural_Breaks as nb
from descartes import PolygonPatch
import fiona
from itertools import chain
import seaborn as sb
from pyproj import Proj, transform
import matplotlib

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
    df = df[df.State == 'MD']
    df = df.reset_index(drop=True)
    return df
    
def defineLargeEmployers(df, YEAR, EAGB_INDUSTRY):
    industry_df = df[(df.FirstYear <= YEAR)&(df.LastYear >= YEAR)][((df.LastMove<=YEAR)&(df.DestState_r=='MD'))|(df.LastMove>YEAR)|df.LastMove.isnull()]
    sortby = 'Emp'+str(YEAR)[2:4]
    cutoff = df.sort(sortby, ascending=False).head(25).tail(1)[sortby]
    cutoff = int(cutoff)
    df = df[df['%s' % sortby] >= cutoff]
    df = df.reset_index(drop=True)
    return df

def GeographySelector(LOCATION="Maryland"):
    if LOCATION == "Maryland":
        # Open Maryland shapefile and extract data
        # used for setting up basemap
        shp = fiona.open(r"G:\Publications\2015\Regional Clustering\Maryland Counties\county\county.shp")
        bds = shp.bounds
        shp.close()
        # Convert bounds to lat/lon
        ll = (bds[0], bds[1])
        ur = (bds[2], bds[3])
        extra = 0.01
        coords = list(chain(ll,ur))
        w, h = coords[2] - coords[0], coords[3]-coords[1]
        lowleftlon = coords[0] - extra * w
        lowleftlat = coords[1] - extra + 0.01 * h
        uprightlon = coords[2] + extra * w
        uprightlat = coords[3] + extra + 0.01 * h
        
        #Create basemap
        m = Basemap(
            projection = "tmerc",
            lon_0=-76.464844,                           # Central longitude
            lat_0=38.22955,                             # Central latitude
            ellps = "WGS84",
            llcrnrlon=coords[0] - extra * w,            # Longitude for lower and upper
            llcrnrlat=coords[1] - extra + 0.01 * h,     # corners of map, given by
            urcrnrlon=coords[2] + extra * w,            # extremes of shapefile +/-
            urcrnrlat=coords[3] + extra + 0.01 * h,     # some extra space
            lat_ts = 0,
            resolution = "i",
            suppress_ticks = True)

        # Need a shapefile with county, ZIP, and/or neighborhood data - DBED ESRI
        m.readshapefile(
            # path to shapefile components
            r"G:\Publications\2015\Regional Clustering\Maryland Counties\county\county",
            # name
            'counties',
            color = 'none',
            zorder=2)

        # Set up a map dataframe of counties and neighborhoods
        # counties = name assigned in m.readshapefile()
        # COUNTY = county key            
        df_map = pd.DataFrame({
            "poly": [Polygon(xy) for xy in m.counties],
            "county_name": [counties["geodesc"] for counties in m.counties_info]})
        df_map["area_m"] = df_map["poly"].map(lambda x: x.area)
        
        # Draw countypatches from polygons
        df_map["patches"] = df_map["poly"].map(lambda x: PolygonPatch(
            x,
            fc="#555555",
            ec="#787878", lw=0.25, alpha=0.9,
            zorder=3))
            
    elif LOCATION == "Baltimore City":
        # Open Baltimore City shapefile and extract data
        # used for setting up basemap
        shp = fiona.open(r"G:\Publications\2015\Regional Clustering\Baltimore Neighborhood Shapefile\BaltimoreNeighborhoods.shp")
        bds = shp.bounds
        shp.close()
        extra = 0.01
        # Convert bounds to lat/lon
        ll = (bds[0], bds[1])
        ur = (bds[2], bds[3])
        coords = list(chain(ll,ur))
        w, h = coords[2] - coords[0], coords[3]-coords[1]
        
        #Create basemap
        m = Basemap(
            projection = "merc",
            lon_0=-76.464844,                           # Central longitude
            lat_0=38.22955,                             # Central latitude
            ellps = "WGS84",
            llcrnrlon=coords[0] - extra * w,            # Longitude for lower and upper
            llcrnrlat=coords[1] - extra + 0.01 * h,     # corners of map, given by
            urcrnrlon=coords[2] + extra * w,            # extremes of shapefile +/-
            urcrnrlat=coords[3] + extra + 0.01 * h,     # some extra space
            lat_ts = 0,
            resolution = "i",
            suppress_ticks = True)
            
        # Need a shapefile with county, ZIP, and/or neighborhood data - DBED ESRI
        m.readshapefile(
            # path to shapefile components
            r"G:\Publications\2015\Regional Clustering\Baltimore Neighborhood Shapefile\BaltimoreNeighborhoods",
            # name
            'neighborhoods',
            #color = 'none',
            zorder=2)
            
        # Set up a map dataframe of counties and neighborhoods
        # neighborhoods = name assigned in m.readshapefile()
        df_map = pd.DataFrame({
            "poly": [Polygon(xy) for xy in m.neighborhoods],
            "neighborhood_name": [neighborhoods["LABEL"] for neighborhoods in m.neighborhoods_info]})
        df_map["area_m"] = df_map["poly"].map(lambda x: x.area)
        
        # Draw countypatches from polygons
        df_map["patches"] = df_map["poly"].map(lambda x: PolygonPatch(
            x,
            fc="#555555",
            ec="#787878", lw=0.25, alpha=0.9,
            zorder=3))
    else:
        print "Map projections are only supported for Maryland and Baltimore City at this time."
    return m, df_map

def ClusterPoints(YEAR, EAGB_INDUSTRY, LANDMARK = "None", LOCATION="Maryland"): 
    # Import CSV based on EAGB_INDUSTRY
    if EAGB_INDUSTRY == "Cyber Security":
        pass        
        #MD_NETS = pd.read_csv(r"", columns=())
    elif EAGB_INDUSTRY in ['Agricultural Chemicals', 'Drugs', 'Research and Testing Services']:
        MD_NETS = pd.read_csv(r"G:\Publications\2015\Regional Clustering\NETS Industry Group Industries\bio_industrygroups.csv", names=col_names, index_col=False)
    elif EAGB_INDUSTRY == "Advanced Manufacturing":
        pass
        #MD_NETS = pd.read_csv(r"", columns=())
    elif EAGB_INDUSTRY=='Manufacturing':
        MD_NETS = pd.read_csv(r'./NETS Industry Group Industries/manufacturing.csv', header=0, names=col_names[:319], index_col=False)
        MD_NETS['EAGBIndustry'] = 'Manufacturing'
    elif EAGB_INDUSTRY=='Incubators':
        MD_NETS = pd.read_csv(r"C:\ProgramData\MySQL\MySQL Server 5.6\data\nets\incubators.csv", names=col_names, index_col=False)
    elif EAGB_INDUSTRY=='EdTech':
        MD_NETS=pd.read_csv(r'G:\Publications\2015\Regional Clustering\NETS Industry Group Industries\edtech.csv', names=col_names, index_col=False)
    elif EAGB_INDUSTRY=='Government Contracting':
        MD_NETS=pd.read_csv(r'G:\Publications\2015\Regional Clustering\NETS Industry Group Industries\govtcontra.csv', names=col_names, index_col=False)
    else:
        print "This program does not currently support mapping of the %s industry" % EAGB_INDUSTRY

    # Prepare full establishment database
    if  EAGB_INDUSTRY in ['Agricultural Chemicals', 'Drugs', 'Research and Testing Services']:
        MD_NETS['EAGBIndustry'] = MD_NETS['IndustryGroup']
    elif EAGB_INDUSTRY == 'Government Contracting':
        MD_NETS['EAGBIndustry'] = np.where(MD_NETS['IndustryGroup']=='Y', 'Government Contracting', 'Government Contracting')
    elif EAGB_INDUSTRY == 'Incubators':
        MD_NETS['EAGBIndustry'] = np.where(MD_NETS['Company']=='Betamore', 'Incubators', 'Incubators')
    elif EAGB_INDUSTRY=='EdTech':
        MD_NETS['EAGBIndustry'] = 'EdTech'
        
    MD_NETS = transformDataFrame(MD_NETS)
    
    if EAGB_INDUSTRY == 'Incubators':
        pass
    else:
        industry_df = MD_NETS[MD_NETS.EAGBIndustry == EAGB_INDUSTRY].reset_index(drop=True)
    
    df_map = GeographySelector(LOCATION)[1]
    
    # Zip establishment coordinates from dataframe
    All_Zipped = zip(MD_NETS["Longitude"], MD_NETS["Latitude"])
    Origin_Zipped = zip(MD_NETS["OriginLongitude"], MD_NETS["OriginLatitude"])
    Destination_Zipped = zip(MD_NETS["DestLongitude"], MD_NETS["DestLatitude"])

    # Create a list of map coordinates from dataframe longitude and latitude variables              
    map_coords = []
    for i, estab in MD_NETS.iterrows():
        if MD_NETS["FirstYear"][i] <= YEAR and MD_NETS["LastYear"][i] >= YEAR and MD_NETS['EAGBIndustry'][i] == EAGB_INDUSTRY:
            if np.isnan(MD_NETS["LastMove"][i]):
                map_coords.append(list(All_Zipped[i]))
            elif MD_NETS["MoveYear_r"][i] <= YEAR:
                map_coords.append(list(Destination_Zipped[i]))
            else:
                map_coords.append(list(Origin_Zipped[i]))
    map_points = pd.Series([Point(m(mapped_x, mapped_y)) for mapped_x, mapped_y in map_coords])
    #map_points = pd.Series(map_coords)
    estab_points = MultiPoint(list(map_points.values))
    county_polygon = prep(MultiPolygon(list(df_map["poly"].values)))
    # Calculcate points that fall within the LOCATION boundary
    map_estab_points = filter(county_polygon.contains, estab_points)
    
    # Create a list of map coordinates from dataframe longitude and latitude variables
    # for landmark establishments, if passed as an optional argument
    # If Hospital/Incubator is in same column, try to condense. Lots of code repeated.
    if LANDMARK == 'None':
        #lmk_estab_points = []
        map_lmk_points = []
    elif LANDMARK == "Hospitals":
        # Read in hospitals CSV
        hospitals = pd.read_csv(r"G:\Publications\2015\Regional Clustering\NETS Industry Group Industries\hospitals.csv", names=col_names, index_col=False)
        hospitals = transformDataFrame(hospitals)
    
        # Then zip them up
        All_hosp_Zipped = zip(hospitals["Longitude"], hospitals["Latitude"])
        Origin_hosp_Zipped = zip(hospitals["OriginLongitude"], hospitals["OriginLatitude"])
        Destination_hosp_Zipped = zip(hospitals["DestLongitude"], hospitals["DestLatitude"])
        
        lmk_coords = []
        for i, estab in hospitals.iterrows():
            # If establishment opened before map year
            # AND establishment is a hospital
            if hospitals["FirstYear"][i] <= YEAR and hospitals["LastYear"][i] >= YEAR:
                if np.isnan(hospitals["LastMove"][i]):
                    lmk_coords.append(list(All_hosp_Zipped[i]))
                elif hospitals["MoveYear_r"][i] <= YEAR:
                    lmk_coords.append(list(Destination_hosp_Zipped[i]))
                else:
                    lmk_coords.append(list(Origin_hosp_Zipped[i]))
        lmk_points = pd.Series([Point(m(mapped_x, mapped_y)) for mapped_x, mapped_y in lmk_coords])
        l#mk_points = pd.Series(lmk_coords)
        lmk_estab_points = MultiPoint(list(lmk_points.values))
        county_polygon = prep(MultiPolygon(list(df_map["poly"].values)))
        # Calculcate points that fall within the LOCATION boundary
        map_lmk_points = filter(county_polygon.contains, lmk_points)
        
    elif LANDMARK == "Incubators":
        # Read in incubators CSV
        incubators = pd.read_csv(r"G:\Publications\2015\Regional Clustering\NETS Industry Group Industries\incubators.xlsx", sheetname='incubators', names=col_names, index_col=False)
        incubators = transformDataFrame(incubators)
    
        # Then zip them up
        All_incubators_Zipped = zip(incubators["Longitude"], incubators["Latitude"])
        Origin_incubators_Zipped = zip(incubators["OriginLongitude"], incubators["OriginLatitude"])
        Destination_incubators_Zipped = zip(incubators["DestLongitude"], incubators["DestLatitude"])
                
        lmk_coords = []
        for i, estab in incubators.iterrows():
            # If establishment opened before map year
            # AND establishment is a hospital
            if incubators["FirstYear"][i] <= YEAR and incubators["LastYear"][i] >= YEAR:
                if np.isnan(incubators["LastMove"][i]):
                    lmk_coords.append(list(All_incubators_Zipped[i]))
                elif incubators["MoveYear_r"][i] <= YEAR:
                    lmk_coords.append(list(Destination_incubators_Zipped[i]))
                else:
                    lmk_coords.append(list(Origin_incubators_Zipped[i]))
        lmk_points = pd.Series([Point(m(mapped_x, mapped_y)) for mapped_x, mapped_y in lmk_coords])
        l#mk_points = pd.Series(lmk_coords)
        lmk_estab_points = MultiPoint(list(lmk_points.values))
        county_polygon = prep(MultiPolygon(list(df_map["poly"].values)))
        # Calculcate points that fall within the LOCATION boundary
        map_lmk_points = filter(county_polygon.contains, lmk_points)
        
    elif LANDMARK == "Large Employers":
        industry_large_employers = defineLargeEmployers(industry_df, YEAR, EAGB_INDUSTRY) 
        # Zip up large employer dataframe
        All_LE_Zipped = zip(industry_large_employers["Longitude"], industry_large_employers["Latitude"])
        Origin_LE_Zipped = zip(industry_large_employers["OriginLongitude"], industry_large_employers["OriginLatitude"])
        Destination_LE_Zipped = zip(industry_large_employers["DestLongitude"], industry_large_employers["DestLatitude"])
        
        # Get coordinates for the location of each large employer in the map year
        lmk_coords = []
        for i, estab in industry_large_employers.iterrows():
            if industry_large_employers['FirstYear'][i] <= YEAR and industry_large_employers["LastYear"][i] >= YEAR:
                if np.isnan(industry_large_employers["LastMove"][i]):
                    lmk_coords.append(list(All_LE_Zipped[i]))
                elif industry_large_employers["MoveYear_r"][i] <= YEAR:
                    lmk_coords.append(list(Destination_LE_Zipped[i]))
                else:
                    lmk_coords.append(list(Origin_LE_Zipped[i]))
        lmk_points = pd.Series([Point(m(mapped_x, mapped_y)) for mapped_x, mapped_y in lmk_coords])
        l#mk_points = pd.Series(lmk_coords)
        lmk_estab_points = MultiPoint(list(lmk_points.values))
        county_polygon = prep(MultiPolygon(list(df_map["poly"].values)))
        # Calculcate points that fall within the LOCATION boundary
        map_lmk_points = filter(county_polygon.contains, lmk_points)
        
    # Continue adding LANDMARKS to map
    else:
        pass
    return map_estab_points, map_lmk_points
   
def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    """
    This is a convenience function to stop you making off-by-one errors
    Takes a standard colour ramp, and discretizes it,
    then draws a colour bar with correctly aligned labels
    """
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, **kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))
    if labels:
        colorbar.set_ticklabels(labels)
    return colorbar

def cmap_discretize(cmap, N):
    """
    Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)

    """
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in xrange(N + 1)]
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

def IndicatorHeatMap(YEAR, LOCATION="Maryland", INDICATOR="None"):
    # Using a CSV of economic indicators, create a heatmap that
    # shows the data. This will be used under scatter and hexbin maps
    # if an indicator is selected.
    # This can probably be turned into an if/elif/else statement that
    # sets a common variable to a string based on the indicator chosen
    # and passes that variable through the breaks and jenks process.
    df_map = pd.merge(df_map, indicator_list, on="county_name", how="left")
    if INDICATOR == "None":
        pass
    elif INDICATOR == "Median Household Income":
        # Select county or neighborhood MHHI data from the passed year
        breaks = nb(
            df_map[df_map["mhhi"].notnull()].mhhi.values,
            initial=300,
            k=5)
            
        jb = pd.DataFrame({"jenks_bins":breaks.yb}, index=df_map[df_map["mhhi"].notnull()].index)
        df_map = df_map.join(jb)
        df_map.jenks_bins.fillna(-1, inplace=True)

        jenks_labels = ["Median Household Income:\n<= %f" % b for b in breaks.bins]
    elif INDICATOR == "Millennial Population Growth":
        # Select county or neighborhood Millennial population data from the passed year
        pass
    else:
        print "The %s indicator is not yet available in this program." % INDICATOR
    
    ax = fig.add_subplot(111, axisbg = "w", frame_on = False)

    # Change get_cmap color based on INDICATOR    
    cmap = plt.get_cmap("Blues")
    df_map["patches"] = df_map["poly"].map(lambda x: PolygonPatch(x, ec="#555555", lw=.2, alpha=1, zorder=4))
    pc = PatchCollection(df_map["patches"], match_original=True)
    norm = Normalize()
    pc.set_facecolor(cmap(norm(df_map["jenks_bins"].values)))
    ax.add_collection(pc)

    cb = colorbar_index(ncolors=len(jenks_labels), cmap=cmap, shrink=0.5, labels=jenks_labels)
    cb.ax.tick_params(labelsize=8)

    m.drawmapscale(
        -125, 20,
        -125, 20,
        10.,
        barstyle = "fancy", labelstyle = "simple",
        fillcolor1 = "w", fillcolor2 = "w",
        fontcolor = "w",
        zorder=9,
        units = "m",
        fontsize =7)    
        
def ScatterMap(YEAR, EAGB_INDUSTRY, LANDMARK="None", LOCATION="Maryland", INDICATOR="None", SAVE=False, DRAFT=False):
    # Get requested basemap
    m = GeographySelector(LOCATION)[0]
    # Get requested df_map
    df_map = GeographySelector(LOCATION)[1]
    
    df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(
        x,
        fc='#c5c9c7',
        ec='#787878', lw=0.5, alpha=0.9,
        zorder=4))

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg="w", frame_on=False)
    #m.drawstates(zorder = 1)
    #m.drawcounties(zorder = 0)
    #m.drawcoastlines(zorder = 2)
        
    # If INDICATOR is selected, plot on heatmap:
    if INDICATOR != 'None':
        IndicatorHeatMap(INDICATOR)
    else:
        pass

    if LANDMARK == 'None':
        pass
    elif LANDMARK == 'Hospitals':
        cLMK = '#9be5aa'
    elif LANDMARK == "Large Employers":
        cLMK = '#020035'
    elif LANDMARK == 'Incubators':
        cLMK = '#a2bffe'
            
    if EAGB_INDUSTRY == "Cyber Security":
        cESTAB = "#33ccff"        
    elif EAGB_INDUSTRY in ['Agricultural Chemicals', 'Drugs', 'Research and Testing Services']:
        cESTAB = '#072B80' #"#E85959"
    elif EAGB_INDUSTRY == "Advanced Manufacturing":
        cESTAB = "#8ECBAD"
    elif EAGB_INDUSTRY == 'Incubators':
        cESTAB = '#ff0490'
        mESTAB = 'h'
    elif EAGB_INDUSTRY=='EdTech':
        cESTAB = '#05472a'
    elif EAGB_INDUSTRY=='Government Contracting':
        cESTAB = '#155084'
    #continue adding new EAGB_INDUSTRY values and colors
    else:
        cESTAB = 'k'
        print "Please select an EAGB-defined industry from the following list:\nCyber Security\nHealth IT\nAdvanced Manufacturing."
        
    m.scatter(
        # Not getting points on your map? Try making geom.x points the y coordinates.
        [geom.x for geom in ClusterPoints(YEAR, EAGB_INDUSTRY, LANDMARK)[0]],
        [geom.y for geom in ClusterPoints(YEAR, EAGB_INDUSTRY, LANDMARK)[0]],
        #latlon = True,
        zorder = 9,
        marker = 'o',
        lw = 0.25,
        facecolor = cESTAB,
        edgecolor = "w",
        alpha = 0.9,
        antialiased = True,
        label = "%(1)s Establishments, %(2)s" % {"1":EAGB_INDUSTRY, "2":YEAR}
        )
        
    # Add additional LANDMARKS
    if LANDMARK == 'None':
        pass
    elif LANDMARK == "Hospitals" or LANDMARK == "Incubators" or LANDMARK=='Large Employers':              
        lmk_dev = m.scatter(
        # Not getting points on your map? Try making geom.x points the y coordinates.
        [geom.x for geom in ClusterPoints(YEAR, EAGB_INDUSTRY, LANDMARK)[1]],
        [geom.y for geom in ClusterPoints(YEAR, EAGB_INDUSTRY, LANDMARK)[1]],
        #latlon = True,
        zorder = 10,
        marker = "*",
        s=40,
        lw = 0.25,
        facecolor = cLMK,
        edgecolor = 'w',
        alpha = 0.8,
        antialiased = True,
        label = "%(1)s, %(2)s" % {"1":LANDMARK, "2":YEAR}
        )
    else:
        print "Please select Large Employers, Incubators, or Hospitals landmarks."
    
    # Plot counties or neighborhoods by adding the PatchCollection to the axes instance
    ax.add_collection(PatchCollection(df_map['patches'].values, match_original=True))
    
    # Copyright and source data info
    smallprint = ax.text(
        0.02, 0,
        ha="left", va = "bottom",
        size = 10,
        color = "#555555",
        transform = ax.transAxes,
        s = "Total Points: %(1)s\nLandmarks: %(2)s\nEconomic Indicator: %(3)s\nContains NETS Database data\n$\copyright$ EAGB copyright and database right 2015" % {"1":len(ClusterPoints(YEAR, EAGB_INDUSTRY, LANDMARK)[0]), "2":LANDMARK, "3":INDICATOR}
        )
            
    # Change title based on draft or final status    
    if DRAFT == True:
        plt.title('DRAFT\nAll BioHealth Establishments, %(2)s\n%(3)s' % {"1":EAGB_INDUSTRY, "2":YEAR, "3":LOCATION})
    else:
        plt.title("All %(1)s Establishments, %(2)s\n%(3)s" % {"1":EAGB_INDUSTRY, "2":YEAR, "3":LOCATION})
    
    plt.legend(loc=6)
    plt.tight_layout()
    fig.set_size_inches(10,7)
    
    #Passed argument tells program whether to save maps
    if SAVE == True:
        if DRAFT == True:
            plt.savefig(r'./Map Output/%(1)s/%(3)s/DRAFT_All %(1)s %(2)s_%(3)s_scatter.png' % {"1":EAGB_INDUSTRY, "2":YEAR, '3':LANDMARK}, alpha = True, dpi=600)
        else:
            plt.savefig(r'G:/Publications/2015/Regional Clustering/Map Output/%(1)s/%(3)s/EPS/All %(1)s %(2)s_%(3)s_scatter.eps' % {"1":EAGB_INDUSTRY, "2":YEAR, '3':LANDMARK}, alpha = True, dpi=600)
            plt.savefig(r"G:/Publications/2015/Regional Clustering/Map Output/%(1)s/%(3)s/PNG/All %(1)s %(2)s_%(3)s_scatter.png" % {"1":EAGB_INDUSTRY, "2":YEAR, '3':LANDMARK}, alpha = True, dpi=600)
    else:
        pass
    
    plt.show()
            
def HexbinMap(YEAR, EAGB_INDUSTRY, LANDMARK="None", LOCATION="Maryland", INDICATOR="None", SAVE=False, DRAFT=False):
    # Get requested basemap
    m = GeographySelector(LOCATION)[0]
    # Get requested df_map
    df_map = GeographySelector(LOCATION)[1]
    
    df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(
        x,
        fc='#c5c9c7',
        ec='#787878', lw=0.5, alpha=0.9,
        zorder=4))

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg="w", frame_on=False)
    #m.drawstates(zorder = 1)
    #m.drawcounties(zorder = 0)
    #m.drawcoastlines(zorder = 2)
    
    multipoint_list = ClusterPoints(YEAR, EAGB_INDUSTRY, LANDMARK)[0]
    
    # If INDICATOR is selected, plot on heatmap:
    if INDICATOR != 'None':
        IndicatorHeatMap(INDICATOR)

    # Plot counties and neighborhoods by adding PatchCollection to axes instance
    ax.add_collection(PatchCollection(df_map['patches'].values, match_original=True))
    
    if LANDMARK == "Large Employers":
        lmk_dev = m.scatter(
        # Not getting points on your map? Try making geom.x points the y coordinates.
        [geom.x for geom in map_estab_points],
        [geom.y for geom in map_estab_points],
        latlon = True,
        zorder = 11,
        marker = np.where(MD_NETS.LargeEmployer == "y", "*", "o"),
        lw = 0.25,
        facecolor = np.where(MD_NETS.LargeEmployer == "y", (0,0,0,0.8), (1,1,1,0)),
        edgecolor = "w",
        alpha = 0.9,
        antialiased = True,
        )
    elif LANDMARK == "Hospitals" or LANDMARK == "Incubators":
        lmk_dev = m.scatter(
        # Not getting points on your map? Try making geom.x points the y coordinates.
        [geom.x for geom in ClusterPoints(YEAR, EAGB_INDUSTRY, LANDMARK)[1]],
        [geom.y for geom in ClusterPoints(YEAR, EAGB_INDUSTRY, LANDMARK)[1]],
        latlon = True,
        zorder = 11,
        marker = "*",
        lw = 0.25,
        facecolor = (0,0,0,0.8),
        edgecolor = "w",
        alpha = 0.8,
        antialiased = True,
        )
    else:
        pass

    hx = m.hexbin(
        np.array([geom.x for geom in multipoint_list]),
        np.array([geom.y for geom in multipoint_list]),
        gridsize = 125,
        bins = "log",
        mincnt = 1,
        edgecolor = "none",
        alpha = 1.,
        lw = 0.2,
        cmap = plt.get_cmap("Blues"),
        zorder=5
        )
    
    # Copyright and source data info
    smallprint = ax.text(
        0.02, 0,
        ha="left", va = "bottom",
        size = 10,
        color = "#555555",
        transform = ax.transAxes,
        s = "Total Points: %(1)s\nLandmarks: %(2)s\nEconomic Indicator: %(3)s\nContains NETS Database data\n$\copyright$ EAGB copyright and database right 2015" % {"1":len(ClusterPoints(YEAR, EAGB_INDUSTRY, LANDMARK)[0]), "2":LANDMARK, "3":INDICATOR}
        )
        
    '''
    m.drawmapscale(
        GeographySelector()[1][0] + 0.08, GeographySelector()[1][1] + 0.015,
        GeographySelector()[1][0], GeographySelector()[1][1],
        10.,
        barstyle = "fancy", labelstyle = "simple",
        fillcolor1 = "w", fillcolor2 = "#555555",
        fontcolor = "#555555",
        zorder=11,
        units = "m",
        fontsize = 7)
    '''
    if DRAFT==True:
        plt.title('DRAFT\nBioHealth Establishment Density, %(2)s\n%(3)s' % {"1":EAGB_INDUSTRY, "2":YEAR, "3":LOCATION})
    else:
        plt.title("%(1)s Establishment Density, %(2)s\n%(3)s" % {"1":EAGB_INDUSTRY, "2":YEAR, "3":LOCATION})
    plt.tight_layout()
    fig.set_size_inches(10,7)
    
    #Passed argument tells program whether to save maps
    if SAVE == True:
        if DRAFT==True:
            plt.savefig("./Map Output/%(1)s/%(3)s/DRAFT_BioHealth %(2)s_%(3)s_hex.png" % {"1":EAGB_INDUSTRY, "2":YEAR, '3':LANDMARK}, alpha = True, dpi=600)
        else:
            plt.savefig(r"G:/Publications/2015/Regional Clustering/Map Output/%(1)s/%(3)s/EPS/All %(1)s %(2)s_%(3)s_hex.eps" % {"1":EAGB_INDUSTRY, "2":YEAR, '3':LANDMARK}, alpha = True, dpi=600)
            plt.savefig("G:/Publications/2015/Regional Clustering/Map Output/%(1)s/%(3)s/PNG/All %(1)s %(2)s_%(3)s_hex.png" % {"1":EAGB_INDUSTRY, "2":YEAR, '3':LANDMARK}, alpha = True, dpi=600)
    else:
        pass
    
    plt.show()
    
def HeatMap(YEAR, EAGB_INDUSTRY, LANDMARK="None", LOCATION="Maryland", INDICATOR="None", SAVE=False, DRAFT=False):
    # Get basemap and df_map
    m, df_map = GeographySelector()[0], GeographySelector()[1]
    
    # Get establishment points for year and industry with ClusterPoints()
    map_estab_points = ClusterPoints(YEAR, EAGB_INDUSTRY)[0]
    
    #Find the density of establishments for EAGB_INDUSTRY in YEAR for each county
    df_map["count"] = df_map["poly"].map(lambda x: int(len(filter(prep(x).contains, map_estab_points))))
    df_map["density_m"] = df_map["count"]/df_map["area_m"]
    df_map.replace(to_replace={"density_m": {0:np.nan}}, inplace=True)
    
    # Calculate Jenks natural breaks for density
    breaks = nb(
        df_map[df_map["density_m"].notnull()].density_m.values,
        initial = 300,
        # Number of bins to sort counties into:
        k = 5)
        
    # The notnull method lets us match indices when joining
    jb = pd.DataFrame({"jenks_bins": breaks.yb}, index=df_map[df_map["density_m"].notnull()].index)
    df_map = df_map.join(jb)
    df_map.jenks_bins.fillna(-1, inplace=True)
    
    jenks_labels = ["<= %0.1f/m$^2$ (%s counties)" % (b, c) for b,c in zip(
        breaks.bins, breaks.counts)]
    jenks_labels.insert(0, "No %(1)s Establishments (%(2)s counties)" % {"1":EAGB_INDUSTRY, "2":len(df_map[df_map["density_m"].isnull()])})

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg="w", frameon=False)

    # Use a color ramp determined by EAGB_INDUSTRY with an if statement
    # Change "Blues" to some variable
    cmap = plt.get_cmap("Blues")
    # Draw counties with grey outlines
    df_map["patches"] = df_map["poly"].map(lambda x: PolygonPatch(x, ec="#555555", lw=0.2, alpha=1.0, zorder=4))
    pc = PatchCollection(df_map["patches"], match_original=True)
    # Impose color map onto patch collection
    norm = Normalize()
    pc.set_facecolor(cmap(norm(df_map["jenks_bins"].values)))
    ax.add_collection(pc,zorder=5)

    # Add a color bar
    cb = colorbar_index(ncolors=len(jenks_labels), cmap=cmap, shrink=0.5, labels=jenks_labels)
    cb.ax.tick_params(labelsize=8)
    '''
    # Show highest densities in descending order
    highest = "\n".join(
        value[1] for _, value in df_map[(df_map["jenks_bins"] == 4)][:10].sort().iterrows())
    highest = "Most Dense Counties:\n\n" + highest
    
    details = cb.ax.text(
    -1., 0-0.007,
    highest,
    ha="right", va="bottom",
    size = 8,
    color = "#555555")
    '''
    
    # Copyright and source data info
    smallprint = ax.text(
        0.02, 0,
        ha="left", va = "bottom",
        size = 10,
        color = "#555555",
        transform = ax.transAxes,
        s = "Classification Method: Jenks Natural Breaks\nTotal Points: %(1)s\nLandmarks: %(2)s\nEconomic Indicator: %(3)s\nContains NETS Database data\n$\copyright$ EAGB copyright and database right 2015" % {"1":len(ClusterPoints(YEAR, EAGB_INDUSTRY, LANDMARK)[0]), "2":LANDMARK, "3":INDICATOR}
        )
    ''' 
    m.drawmapscale(
        coords[0] + 0.08, coords[1] + 0.015,
        coords[0], coords[1],
        10.,
        barstyle = "fancy", labelstyle = "simple",
        fillcolor1 = "w", fillcolor2 = "#555555",
        fontcolor = "#555555",
        zorder=11,
        units = "m",
        fontsize = 7) 
    ''' 
    plt.tight_layout()
    fig.set_size_inches(10,10)
    plt.title("%(1)s Establishment Density, %(2)s\n%(3)s"), {"1":EAGB_INDUSTRY, "2":YEAR, "3":LOCATION}

    #Passed argument tells program whether to save maps    
    if SAVE == True:
        plt.savefig("All %(1)s %(2)s_heatmap.eps" % {"1":EAGB_INDUSTRY, "2":YEAR}, alpha = True)
        plt.savefig("All %(1)s %(2)s_heatmap.png" % {"1":EAGB_INDUSTRY, "2":YEAR}, alpha = True)
    else:
        pass
    
    plt.show()
        
# Define a single function with multiple levers that will print a map of our choosing
def PrintMap(YEAR, EAGB_INDUSTRY, LANDMARK="None", LOCATION="Maryland", INDICATOR="None", PLOTTYPE="scatter", SAVE=False):
    m, coords = GeographySelector(Location='Maryland')
    if PLOTTYPE == 'scatter:
        ScatterMap(YEAR, EAGB_INDUSTRY, LANDMARK="None", LOCATION="Maryland", INDICATOR="None", SAVE=False)
    elif PLOTTYPE == 'hex':
        HexbinMap(YEAR, EAGB_INDUSTRY, LANDMARK="None", LOCATION="Maryland", INDICATOR="None", SAVE=False)
    elif PLOTTYPE == "heatmap":
        HeatMap(YEAR, EAGB_INDUSTRY, LOCATION="Maryland")
    colorbar_index()
    cmap_discretize()