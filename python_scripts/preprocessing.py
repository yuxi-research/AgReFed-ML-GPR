"""
Preprocessing functions for geospatial soil data 


Requirements:
- python>=3.9
- matplotlib>=3.5.1
- numpy>=1.22.0
- pandas>=1.3.5
- PyYAML>=6.0
- geopandas>=0.7.0

For more package details see conda environment file: environment.yaml

This package is part of the machine learning project developed for the Agricultural Research Federation (AgReFed).

Copyright 2022 Sebastian Haan, Sydney Informatics Hub (SIH), The University of Sydney

This open-source software is released under the AGPL-3.0 License.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import os
import sys
import argparse
import yaml
from types import SimpleNamespace  

# Default settings yaml file name:
_fname_settings = 'settings_preprocessing.yaml'


def preprocess(inpath, infname, outpath, outfname, name_target, name_features, 
             zmin = None, zmax = None, gen_gpkg = False, categorical = None,
            colname_depthmin = 'lower_depth', colname_depthmax = 'upper_depth',
            colname_xcoord = 'x', colname_ycoord = 'y', colname_lat = None, colname_lng= None, 
            project_crs = None): 
    """
    Converts input dataframe into more useable data and saves as new dataframe (csv file) on disk.

    Preprocessing steps:
    
    - Cleaning data
    - Tries to finds Latitude and Longitude entries
    - Calculation of depth intervals and their mid-points.
    - Trims data to zmin and zmax
    - Automatically converts categorical features to binary feature representations.
    - Generates cross-validation indices for cross-validation.
    
    - Generates geospatial dataframe with coordinates (not yet implemented)


    Input:
        inpath: input path
        infname: input file name
        outpath: output path
        outfname: output file name
        name_target: String, Name of target for prediction (column names in infname)
        name_features: String or list of strings of covariate features (column names in infname)
        zmin: in centimeters, if not None, data only larger than zmin will be selected
        zmax: in centimeters, if not None, data only smaller than zmax will be selected
        gen_gpkg: if True, data will be also saved as georeferenced geopackage
        categorical: name of categorical feature, converts categorical feature into additional features with binary coding
        colname_depthmin: name of column for lower depth
        colname_depthmax: name of column for upper depth
        colname_xcoord: name of column for Easting coordinate (if projected crs this is in meters)
        colname_ycoord: name of column for Northing coordinate (if projected crs this is in meters)
        project_crs: string, EPSG code for projected coordinate reference system 
        of colname_xcoord and colname_ycoord (e.g. 'EPSG:28355')

    TODO: automatically detect if coordinates are projected or not
    """
    ## Keep track of all covariates
    #name_features2 = name_features.copy()
    # Keep track of all relevant column names
    fieldnames = name_features + [name_target]
    #df = pd.read_csv(os.path.join(inpath, infname), usecols = fieldnames)
    df = pd.read_csv(os.path.join(inpath, infname))
    # Find if Latitude or Longitude values exist:
    if ('Latitude' in list(df)) &  ('Longitude' in list(df)):
        fieldnames += ['Latitude', 'Longitude']
    else:
        if ('Lat' in list(df)):
            df.rename(columns={"Lat": "Latitude"}, inplace=True)
            fieldnames += ['Latitude']
        if ('Lng' in list(df)):
            df.rename(columns={"Lng": "Longitude"}, inplace=True)
            fieldnames += ['Longitude']
        elif ('Lon' in list(df)):
            df.rename(columns={"Lon": "Longitude"}, inplace=True)
            fieldnames += ['Longitude']
    # Check if x and y are in meters (projected coordinates) or in degrees:
    # Here we assume that bounding box is not larger than 5 degree in any directions
    if (abs(df.x.max() - df.x.min()) < 5) | (abs(df.y.max() - df.y.min()) < 5):
        print(f'WARNING: Coordinates {colname_xcoord} and {colname_ycoord} seem to be not in meters!')
        print(f'         Please check if coordinates are projected or not!')

    # Calculate mid point and convert to cm
    df['z'] = 0.5 * (df[colname_depthmin] + df[colname_depthmax]) / 100.
    df['z_diff'] = (df[colname_depthmin] - df[colname_depthmax]) / 100.
    fieldnames += ['z', 'z_diff']
    #if ('Easting' in list(df)) &  ('Northing' in list(df)) & ('x' not in list(df)) & ('y' not in list(df)):
    #    df.rename(columns={"Easting": "x", "Northing": "y"}, inplace=True)
    if ('x' not in list(df)) & ('y' not in list(df)):
        df.rename(columns={colname_xcoord: "x", colname_ycoord: "y"}, inplace=True)  
        fieldnames += ['x', 'y']  
    #if isinstance(name_features2, list):
    #    selcols = ['x', 'y', 'z', 'z_diff']
    #    selcols.extend(name_features2)
    #else:
    #    selcols = ['x', 'y', 'z', 'z_diff', name_features2]
    #selcols.extend([name_target])

    # Trim data to zmin and zmax
    if zmin is not None:
        df = df[df.z >= zmin]
    if zmax is not None:
        df = df[df.z <= zmax]

    # Continue only with relevant fields
    df = df[fieldnames]

    # Find categorical features in dataframe (here we assume all strings are categorical)
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    #names_categorical  df.select_dtypes(exclude=['number','datetime']).columns.tolist()
    # Convert categorical features to binary features
    if len(categorical) > 0:
        # Add categories as features with binary coding
        for name_categorical in categorical:
            cat_levels = df[name_categorical].unique()
            print('Adding following categories as binary features: ', cat_levels)
            fieldnames.extend(cat_levels)
            for level in cat_levels:
                df[level] = 0
                df.loc[df[name_categorical].values == level, level] = 1
            fieldnames.remove(name_categorical)


    #Keep only finite values (remove nan, inf, -inf)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace = True)
    

    """
    # Generate cross-validation indices ()
    if name_ixval:
        ### Splitting in N-fold cross-validation
        if name_ixval not in list(df):
            print('Creating 10-fold cross-validation samples...')
            nsplit = 10
            df['site_id'] = df['x'].round(0).astype(int).map(str) + '_' + df['y'].round(0).astype(int).map(str)
            nunique = df['site_id'].unique()
            df['new_id'] = 0
            ix = 1
            for unique in nunique:
                df.loc[df['site_id'] == unique, 'new_id'] = ix
                ix += 1
            size = int(len(nunique)/nsplit)

            np.random.shuffle(nunique)
            df[name_ixval] = 0
            start = 0
            for i in range(nsplit - 1):
                stop = start + size
                sub = nunique[start:stop]
                df.loc[df['site_id'].isin(sub),name_ixval] = i + 1
                start = stop
            df.loc[df[name_ixval] == 0, name_ixval] = nsplit
        selcols.extend([name_ixval])
        #dfout = dfout[selcols].copy()
    dfout.to_csv(os.path.join(outpath, outfname), index = False)
    """

    # Finally save dataframe as csv
    df.to_csv(os.path.join(outpath, outfname), index=False)



    # save also as geopackage for visualisation  etc:
    if gen_gpkg:
        if project_crs is not None:
            gdf = gpd.GeoDataFrame(df.copy(), 
                geometry=gpd.points_from_xy(df['x'], df['y']), 
                crs = project_crs).to_file(os.path.join(outpath, outfname + '.gpkg'), driver='GPKG')
        elif ('Latitude' in df) & ('Longitude' in df):
            lat_cen = df.Latitude.mean()
            lng_cen = df.Longitude.mean()
            # find zone from center point
            #zone, crs = find_zone(lat_cen, lng_cen) # could add this function in future
            crs_epsg = 'EPSG:' + str(crs)
            # Use general Australian projection for now: EPSG:8059 (GDA2020 / SA Lambert for entire Australia)
            # Save as non-projected withe Latitude and Longitude
            gdf = gpd.GeoDataFrame(df.copy(), 
                geometry=gpd.points_from_xy(df['Longitude'], dfout['Latitude']), 
                crs='EPSG:4326').to_file(os.path.join(outpath, outfname + '.gpkg'), driver='GPKG')
        else:
            print('WARNING: Cannot generate geopackage file!')
            print('         Please check to provide either project crs or include Latitude/Longitude in data!')




def preprocess_grid(inpath, infname, outpath, outfname, name_features, categorical = None):
    """
    Converts input dataframe into more useable data and saves as new dataframe on disk
    categorical variables are converted into binary features

    Input:
        inpath: input path
        infname: input file name
        outpath: output path
        outfname: output file name
        name_features: String or list of strings of covariate features (column names in infname)
        categorical: name of categorical feature, converts categorical feature into additional features with binary coding

    Return:
        processed dataframe
        names of features with updated categorical (if none, then return original input name_features)
    """
    print("Preprocessing grid covariates...")
    name_features2 = name_features.copy()
    df = pd.read_csv(os.path.join(path, infname))
    if categorical is not None:
        # Add categories as features with binary coding
        cat_levels = df[categorical].unique()
        print('Adding following categories as binary features: ', cat_levels)
        name_features2.extend(cat_levels)
        for level in cat_levels:
            df[level] = 0
            df.loc[df[categorical].values == level, level] = 1
        name_features2.remove(categorical)
    if ('Easting' in list(df)) &  ('Northing' in list(df)) & ('x' not in list(df)) & ('y' not in list(df)):
        df.rename(columns={"Easting": "x", "Northing": "y"}, inplace=True)
    if isinstance(name_features2, list): 
        selcols = ['x', 'y']
        selcols.extend(name_features2)
    else:
        selcols = ['x', 'y', name_features2]
    dfout = df[selcols].copy()
    dfout.to_csv(os.path.join(outpath, outfname), index = False)   
    return dfout, name_features2


def preprocess_grid_poly(path, infname_grid, infname_poly, name_features, 
	grid_crs, grid_colname_Easting = 'x', grid_colname_Northing = 'y',
	categorical = None):
	"""
	Converts input dataframe into more useable data and saves as new dataframe on disk.
	Grid points will be spatially joioned with polygons for prediction
	To Do add conversion to categorical

	Input:
        path: input path (same used for output)
        infname_grid: input file name of grid covariates
        infname_poly: input file name of polygon shape for predictions
        name_features: String or list of strings of covariate features (column names in infname)
        grid_crs: coordinate reference system of grid points, e.g. 'EPSG:28355'
        grid_colname_Easting: column name of Easting coordinate (or Longitude)
        grid_colname_Northing: coumn name for Northing coodinate (or Latitude)
        categorical: name of categorical feature, converts categorical feature into additional features with binary coding
	
    Return:
        processed dataframe
        names of features with updated categorical (if none, then return original input name_features)
	"""
	from shapely.geometry import Point
	print("Preprocessing grid covariates and joining with polygon geometry...")
	name_features2 = name_features.copy()
	df = pd.read_csv(os.path.join(path, infname_grid))
	if categorical is not None:
		# Add categories as features with binary coding
		cat_levels = df[categorical].unique()
		print('Adding following categories as binary features: ', cat_levels)
		name_features2.extend(cat_levels)
		for level in cat_levels:
			df[level] = 0
			df.loc[df[categorical].values == level, level] = 1
		name_features2.remove(categorical)
	# Convert grid covariates to geospatial point data
	geometry = [Point(xy) for xy in zip(df[grid_colname_Easting], df[grid_colname_Northing])]
	gdf = gpd.GeoDataFrame(df, crs=grid_crs, geometry=geometry)
	# Read in polygon data
	dfpoly =  gpd.read_file(os.path.join(path, infname_poly))
	dfpoly['ibatch'] = dfpoly.index.values.astype(int)
	# Check before merging that grid and poly are in same crs, of not, convert poly to same crs as grid
	if not (dfpoly.crs == gdf.crs):
		dfpoly = dfpoly.to_crs(gdf.crs)
	# Spatially merge points that are within polygons and keep point grid geometry
	dfcomb = gpd.sjoin(gdf, dfpoly, how="left", op="within") 
	# alternatively use op='intersects' to includfe also points that are on boundary of polygon
	# Remove all points that are not in a polygon
	dfcomb.dropna(subset = ['ibatch'], inplace=True)
	dfcomb['ibatch'] = dfcomb['ibatch'].astype(int) # Need to be converted again to int since merge changes type
	if len(dfcomb) == 0:
		print('WARNING: NO GRID POINTS IN POLYGONS!')
	
	# if convert_to_crs is not None:
	# 	# Convert to meters and get batch x,y coordinate list for each polygon:
	# 	dfcomb = dfcomb.to_crs(epsgcrs)
	# 	dfpoly = dfpoly.to_crs(epsgcrs)
	
	""" If evalutaion points are not given by grid points, following approaches can be tried
	evaluation  points need to subdivide polygon in equal areas (voronoi), otherwise not equal spatial weight 
	options a) rasterize polygon  and then turn pixels in points, b) use pygridtools https://github.com/Geosyntec/pygridtools
	c) use Centroidal Voronoi tessellation (use vorbin or pytess), d) use iterative inwards buffering, e) find points by dynmaic simulation
	or split in halfs iteratively: https://snorfalorpagus.net/blog/2016/03/13/splitting-large-polygons-for-faster-intersections/
	use centroid, then any line through centroid splits polygon in half
	Porpose new algorithm: centroidal hexagonal grid (densest cirle packing), 
	this need optimise only one parameter: rotational angle so that a) maximal number of poinst are in polygon and 
	b) that distance of point to nearest polygon edge is maximised for all points (which aligns points towards center)
	"""
	if isinstance(name_features2, list): 
		selcols = ['x', 'y', 'ibatch']
		selcols.extend(name_features2)
	else:
		selcols = ['Sample.ID','x', 'y', 'ibatch', name_features2]
	return dfcomb[selcols].copy(), dfpoly, name_features2


def main(fname_settings):
    """
    Main function for running the script.

    Input:
        fname_settings: path and filename to settings file
    """
    # Load settings from yaml file
    with open(fname_settings, 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    # Parse settings dictinary as namespace (settings are available as 
    # settings.variable_name rather than settings['variable_name'])
    settings = SimpleNamespace(**settings)

    # Verify output directory and make it if it does not exist
    os.makedirs(settings.outpath, exist_ok = True)

    # Preprocess data
    preprocess(settings.inpath, settings.infname, 
            settings.outpath, settings.outfname, 
            settings.name_target, settings.name_features, 
            zmin = 100*settings.zmin, zmax= 100*settings.zmax, 
            categorical = 'Soiltype',
            colname_depthmin = settings.colname_depthmin, colname_depthmax = settings.colname_depthmax)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess data for machine learning on soil data.')
    parser.add_argument('-s', '--settings', type=str, required=False,
                        help='Path and filename of settings file.',
                        default = _fname_settings)
    args = parser.parse_args()

    # Run main function
    main(args.settings)