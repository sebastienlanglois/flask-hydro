import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import fiona
import glob
import multiprocessing as mp
import os
import sqlite3
import json
import geopandas as gpd
from flask import url_for

def clip_raster_classes_to_dict(raster_path, feature, var_physio_list):
    """ Opens a raster file from raster_path and applies a mask based on
        a polygon (feature). It then extracts the percentage of every class
        with respects to the total number of pixels contained in the mask.

    :param raster_path: raster path (raster must contain classes)
    :param feature: polygon feature (extracted from a shapefile or geojson)
    :param var_physio_list: dictionary with class's number as keys and name as value
    :return: dictionary containing the percentage of pixels for each class contained
             in the mask
    """
    with rasterio.open(raster_path) as src:
        # Apply mask to raster and crop
        out_image, out_transform = rasterio.mask.mask(src,
                                                      [feature["geometry"]], crop=True)
        # Count number of pixel for each class contained in mask
        unique_class, counts = np.unique(out_image[0], return_counts=True)

        # Make a dictionary and remove classes > 8 (such as 255)
        dict_unique_class_counts = dict(zip(list(var_physio_list.values()),
                                            [counts[unique_class == class_value][0]
                                             if class_value in unique_class[unique_class <= 8]
                                             else 0 for class_value in list(var_physio_list.keys())]))

        # Count total pixel and get percentage for each class
        total_nb_px = sum(dict_unique_class_counts.values())
        dict_unique_class_counts.update((key, value/total_nb_px)
                                        for key, value in dict_unique_class_counts.items())
    return dict_unique_class_counts


def terrain_analysis_to_dict(feature):
    """

    :param feature:
    :return:
    """

    poly = gpd.GeoDataFrame.from_features([feature])["geometry"]
    poly.crs = {'init': 'epsg:4326'}
    poly_proj = poly.to_crs(epsg=6622)
    area = poly_proj.area[0]/1000000
    perimeter = poly_proj.length[0]/1000
    rayon_cercle_equiv = np.sqrt(area/np.pi)
    gravelius = (perimeter/(2*np.pi*rayon_cercle_equiv))
    rayon_cercle_equiv = perimeter / (2 * np.pi)
    circularite = (area/(np.pi*rayon_cercle_equiv**2))

    return {
            "GRAV": gravelius,
            "RCIR": circularite,
            "SUBV": area,
            "PERI": perimeter,
        }


def clip_raster_mean(raster_path, feature, var_nam):
    """ Opens a raster file from raster_path and applies a mask based on
        a polygon (feature). It then extracts the percentage of every class
        with respects to the total number of pixels contained in the mask.

    :param   raster_path: raster path (raster must contain classes)
    :param   feature: polygon feature (extracted from a shapefile or geojson)
    :return: dictionary containing the percentage of pixels contained
             in the mask
    """
    with rasterio.open(raster_path) as src:
        # Apply mask to raster and crop
        out_image, out_transform = rasterio.mask.mask(src,
                                                      [feature["geometry"]], crop=True)
        if var_nam == 'PTED':
            out_image[out_image < 0] = np.nan

    return np.nanmean(out_image)


def create_dataframe(value, feature, type_requete, source, var_name, id_name):
    """
     Merge dict_unique_class_counts and feature to a pandas dataframe
        which contains data on the relative amount of each class in the feature

    :param value: dictionary or list containing the percentage of pixels
           for each class contained in the mask
    :param feature: feature: polygon feature (extracted from a shapefile or geojson)
    :param type_requete: type of request
    :param source: data origin
    :param var_name: variable name provided if not land cover
    :return: Pandas dataframe with relative amount of each class in the feature
    """

    # feature must have a name in STATION property
    try:
        if id_name is not None:
            station_info = feature['properties'][id_name]
        elif 'Station' in feature['properties']:
            station_info = feature['properties']['Station']
        elif 'STATION' in feature['properties']:
            station_info = feature['properties']['STATION']
        elif 'NOM' in feature['properties']:
            station_info = feature['properties']['NOM']

        # valid for CEHQ shapefile format
        if "ST_" in station_info:
            station_name = station_info.split('_', 1)[1]
        elif "_" in station_info:
            station_name = station_info.split('_', 1)[0]
        else:
            station_name = station_info

        # Create dataframe from dict or list
        if type_requete == 'OCC_SOL':
            # Append PFOR variable
            df = pd.DataFrame.from_dict(value, orient='index', columns=['VALUE'])
            df = df.append(
                pd.DataFrame({'VARIABLE': 'PFOR', 'VALUE':
                             df[df.index.isin(['PCON', 'PFEU', 'PMIX'])]
                             .sum()}).set_index('VARIABLE'))
            df['VARIABLE'] = df.index
        elif type_requete == 'TERRAIN':
            df = pd.DataFrame.from_dict(value, orient='index', columns=['VALUE'])
            df['VARIABLE'] = df.index
        else:
            df = pd.DataFrame([value], columns=['VALUE'])
            df['VARIABLE'] = var_name
        # Add other columns to dataframe

        df['TYPE'] = 'PHYSIO'
        df['SOURCE'] = source
        df['NUMERO_STATION'] = station_name

        df = df.set_index('NUMERO_STATION')

    except Exception as e:
        df = id_name

    return df


def apply_func(args):
    """ Pure function witch calls clip_raster_classes_to_dict
        and create_dataframe for a unique feature. Intended for
        parallelism

    :param args :raster_path: raster path (raster must contain classes)
    :param args :shp_path: shapefile or geojson path
    :param args :var_physio_list: dictionary with class's number as keys and name as value
    :return: SQL-insert ready pandas dataframe (only land use data)
    """
    raster_path, shp_path, var_physio_list, type_requete, source, var_name, id_name = args
    print(os.path.basename(shp_path))

    list_df = []
    with fiona.open(shp_path, "r", encoding='utf-8') as shapefile:
        features = [feature for feature in shapefile]

        for feature in features:
            if type_requete == "OCC_SOL":
                unique_class_counts = clip_raster_classes_to_dict(raster_path, feature, var_physio_list)
            elif type_requete == "TERRAIN":
                unique_class_counts = terrain_analysis_to_dict(feature)
            else:
                unique_class_counts = clip_raster_mean(raster_path, feature, var_name)
            list_df.append(
                create_dataframe(unique_class_counts, feature, type_requete, source, var_name, id_name))

    return pd.concat(list_df)


def create_occ_sol_from_raster(shp_path, raster_path, var_physio_list,
                               type_requete, source, var_name, nb_processes, id_name=None):
    """
    Loop on all features in shp_path, apply asynchronous function in parallel
    and get result back in a pandas dataframe

    :param shp_path: : shapefile or geojson path
    :param raster_path: raster path (raster must contain classes)
    :param var_physio_list: dictionary with class's number as keys and name as value
    :param type_requete: land cover or single value
    :param source: data origin
    :param var_name: variable name if single value, empty if land cover
    :param nb_processes: number of processes
    :param id_name: unique id column name
    :return: SQL-insert ready pandas dataframe (only land use data)
    """

    lst_shp = []
    for dirpath, dirnames, filenames in os.walk(shp_path):
        for filename in [f for f in filenames if f.endswith(".shp")]:
            lst_shp.append(os.path.join(dirpath, filename))

    # Initiate pool
    pool = mp.Pool(processes=nb_processes)
    list_df = pool.map(apply_func, [(raster_path, shp, var_physio_list, type_requete, source, var_name, id_name)
                                    for shp in lst_shp])

    return pd.concat(list_df)


def create_physio(SHP_PATH,DB_PATH, OCC_SOL_100m_PATH, PENTE_250m_PATH, ALT_250m_PATH, options, nb_proc):
    """

    :param SHP_PATH:
    :param DB_PATH:
    :param OCC_SOL_100m_PATH:
    :param PENTE_250m_PATH:
    :param ALT_250m_PATH:
    :param options:
    :return:
    """

    df_occ = create_occ_sol_from_raster(shp_path=SHP_PATH, raster_path=OCC_SOL_100m_PATH,
                                        var_physio_list=options, type_requete="OCC_SOL",
                                        source="HQP_100m_MODIS", var_name="",
                                        nb_processes=nb_proc)
    df_pente = create_occ_sol_from_raster(shp_path=SHP_PATH, raster_path=PENTE_250m_PATH,
                                        var_physio_list="", type_requete="",
                                        source="GTDEM_250m", var_name="PTED",
                                        nb_processes=nb_proc)
    df_alt = create_occ_sol_from_raster(shp_path=SHP_PATH, raster_path=ALT_250m_PATH,
                                        var_physio_list="", type_requete="",
                                        source="GTDEM_250m", var_name="ALTM",
                                        nb_processes=nb_proc)


    df = pd.concat([df_occ, df_pente, df_alt])
    conn = sqlite3.connect(DB_PATH)
    sql = """ SELECT NUMERO_STATION, ID_POINT , SUPERFICIE, GEOM FROM META_STATION_BASSIN"""
    meta_sta_hydro = pd.read_sql_query(sql, conn)

    df = pd.merge(df, meta_sta_hydro[['NUMERO_STATION', 'ID_POINT']],
                  left_on='NUMERO_STATION', right_on='NUMERO_STATION',
                  how='left').drop(columns=['NUMERO_STATION'])
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df = df.sort_values(by=['ID_POINT'])

    for id_point in df['ID_POINT'].unique():
        print(id_point)
        geom = meta_sta_hydro[meta_sta_hydro['ID_POINT'] == id_point]['GEOM'].values[0]
        poly = gpd.GeoDataFrame.from_features(json.loads(geom))["geometry"]
        poly.crs = {'init': 'epsg:4326'}
        poly_proj = poly.to_crs(epsg=6622)
        area = meta_sta_hydro[meta_sta_hydro['ID_POINT'] == id_point]['SUPERFICIE'].values[0]
        if area == 0:
            area = poly_proj.area[0]/1000000
        perimeter = poly_proj.length[0]/1000
        rayon_cercle_equiv = np.sqrt(area/np.pi)
        gravelius = (perimeter/(2*np.pi*rayon_cercle_equiv))
        rayon_cercle_equiv = perimeter / (2 * np.pi)
        circularite = (area/(np.pi*rayon_cercle_equiv**2))

        # Add values
        df = df.append({'ID_POINT': id_point, 'SOURCE': 'HQP', 'TYPE': 'PHYSIO',
                        'VALUE': gravelius, 'VARIABLE': 'GRAV'}, ignore_index=True)
        df = df.append({'ID_POINT': id_point, 'SOURCE': 'HQP', 'TYPE': 'PHYSIO',
                        'VALUE': circularite, 'VARIABLE': 'RCIR'}, ignore_index=True)
        df = df.append({'ID_POINT': id_point, 'SOURCE': 'HQP', 'TYPE': 'PHYSIO',
                        'VALUE': area, 'VARIABLE': 'SUBV'}, ignore_index=True)
        df = df.append({'ID_POINT': id_point, 'SOURCE': 'HQP', 'TYPE': 'PHYSIO',
                        'VALUE': perimeter, 'VARIABLE': 'PERI'}, ignore_index=True)

        for source in ["VDM/Interpoleau"]:  #, "ERA5"]:
            # TMAX

            sql = """SELECT DATE, VALUE FROM DON_TS
                    WHERE ID_SERIE in (
                        SELECT  ID_SERIE
                        FROM 
                            META_TS
                        WHERE ID_POINT = "%s"
                        AND TYPE_SERIE = "Tmax"
                        AND PAS_DE_TEMPS = "1_M"
                        AND SOURCE = "%s"
                        AND AGGREGATION = "moy")  
                    ORDER BY DATE
                  """% (id_point, source)

            tmax = pd.read_sql_query(sql, conn, index_col="DATE")
            tmax.index = pd.to_datetime(tmax.index)
            agg = tmax.groupby(tmax.index.strftime('%m'))['VALUE'].mean().sort_index(axis=0)
            for idx, value in agg.items():
                df = df.append({'ID_POINT': id_point, 'SOURCE': source, 'TYPE': 'METEO',
                                'VALUE': value, 'VARIABLE': 'TMAX_' + str(idx)}, ignore_index=True)

            # TMIN

            sql = """SELECT DATE, VALUE FROM DON_TS
                    WHERE ID_SERIE in (
                        SELECT  ID_SERIE
                        FROM 
                            META_TS
                        WHERE ID_POINT = "%s"
                        AND TYPE_SERIE = "Tmin"
                        AND PAS_DE_TEMPS = "1_M"
                        AND SOURCE = "%s"
                        AND AGGREGATION = "moy")  
                    ORDER BY DATE
                  """% (id_point, source)
            tmin = pd.read_sql_query(sql, conn, index_col="DATE")
            tmin.index = pd.to_datetime(tmin.index)
            agg = tmin.groupby(tmin.index.strftime('%m'))['VALUE'].mean().sort_index(axis=0)
            for idx, value in agg.items():
                df = df.append({'ID_POINT': id_point, 'SOURCE': source, 'TYPE': 'METEO',
                                'VALUE': value, 'VARIABLE': 'TMIN_' + str(idx)}, ignore_index=True)

            # DJBZ

            sql = """SELECT DATE, VALUE FROM DON_TS
                    WHERE ID_SERIE in (
                        SELECT  ID_SERIE
                        FROM 
                            META_TS
                        WHERE ID_POINT = "%s"
                        AND TYPE_SERIE = "Temperature"
                        AND PAS_DE_TEMPS = "1_J"
                        AND SOURCE = "%s"
                        AND AGGREGATION = "moy")  
                    ORDER BY DATE
                  """% (id_point, source)
            djbz = pd.read_sql_query(sql, conn, index_col="DATE")
            djbz.index = pd.to_datetime(djbz.index, utc=True)
            agg = djbz.groupby(djbz.index.strftime('%j'))['VALUE'].mean().sort_index(axis=0)
            agg[agg > 0] = 0
            value = abs(agg).sum()
            df = df.append({'ID_POINT': id_point, 'SOURCE': source, 'TYPE': 'METEO',
                            'VALUE': value, 'VARIABLE': 'DJBZ'}, ignore_index=True)

            # PRCP

            sql = """SELECT DATE, VALUE FROM DON_TS
                    WHERE ID_SERIE in (
                        SELECT  ID_SERIE
                        FROM 
                            META_TS
                        WHERE ID_POINT = "%s"
                        AND TYPE_SERIE = "Precipitation"
                        AND PAS_DE_TEMPS = "1_M"
                        AND SOURCE = "%s"
                        AND AGGREGATION = "somme")  
                    ORDER BY DATE
                  """% (id_point, source)
            prcp = pd.read_sql_query(sql, conn, index_col="DATE")
            prcp.index = pd.to_datetime(prcp.index)
            agg = prcp.groupby(prcp.index.strftime('%m'))['VALUE'].mean().sort_index(axis=0)
            for idx, value in agg.items():
                df = df.append({'ID_POINT': id_point, 'SOURCE': 'HQP', 'TYPE': 'METEO',
                                'VALUE': value, 'VARIABLE': 'PRCP_' + str(idx)}, ignore_index=True)

            # PRCP_11_3
            value = agg[[10, 11, 0, 1, 2]].sum()
            df = df.append({'ID_POINT': id_point, 'SOURCE': source, 'TYPE': 'METEO',
                            'VALUE': value, 'VARIABLE': 'PRCP_11_3'}, ignore_index=True)
            # PRCP_11_4
            value = agg[[10, 11, 0, 1, 2, 3]].sum()
            df = df.append({'ID_POINT': id_point, 'SOURCE': source, 'TYPE': 'METEO',
                            'VALUE': value, 'VARIABLE': 'PRCP_11_4'}, ignore_index=True)
            # PRCP_12_3
            value = agg[[11, 0, 1, 2]].sum()
            df = df.append({'ID_POINT': id_point, 'SOURCE': source, 'TYPE': 'METEO',
                            'VALUE': value, 'VARIABLE': 'PRCP_12_3'}, ignore_index=True)
            # PRCP_12_4
            value = agg[[11, 0, 1, 2, 3]].sum()
            df = df.append({'ID_POINT': id_point, 'SOURCE': source, 'TYPE': 'METEO',
                            'VALUE': value, 'VARIABLE': 'PRCP_12_4'}, ignore_index=True)

            # PTMA
            value = agg.sum()
            df = df.append({'ID_POINT': id_point, 'SOURCE': source, 'TYPE': 'METEO',
                            'VALUE': value, 'VARIABLE': 'PTMA'}, ignore_index=True)


            sql = """SELECT DATE, VALUE FROM DON_TS
                    WHERE ID_SERIE in (
                        SELECT  ID_SERIE
                        FROM 
                            META_TS
                        WHERE ID_POINT = "%s"
                        AND TYPE_SERIE = "Pluie"
                        AND PAS_DE_TEMPS = "1_M"
                        AND SOURCE = "%s"
                        AND AGGREGATION = "somme")  
                    ORDER BY DATE
                  """% (id_point, source)
            prcp = pd.read_sql_query(sql, conn, index_col="DATE")
            prcp.index = pd.to_datetime(prcp.index)
            agg = prcp.groupby(prcp.index.strftime('%m'))['VALUE'].mean().sort_index(axis=0)

            # PLME
            value = agg[[6, 7, 8, 9, 10, 11]].sum()
            df = df.append({'ID_POINT': id_point, 'SOURCE': source, 'TYPE': 'METEO',
                            'VALUE': value, 'VARIABLE': 'PLME'}, ignore_index=True)

        # EEN (seulement ERA5)

        sql = """SELECT DATE, VALUE FROM DON_TS
                WHERE ID_SERIE in (
                    SELECT  ID_SERIE
                    FROM 
                        META_TS
                    WHERE ID_POINT = "%s"
                    AND TYPE_SERIE = "EEN"
                    AND PAS_DE_TEMPS = "1_M"
                    AND SOURCE = "ERA5"
                    AND AGGREGATION = "moy")  
                ORDER BY DATE
              """% (id_point)

        prcp = pd.read_sql_query(sql, conn, index_col="DATE")
        prcp.index = pd.to_datetime(prcp.index)
        # EEN_1
        value = agg[[0]].mean()
        df = df.append({'ID_POINT': id_point, 'SOURCE': "ERA5", 'TYPE': 'PHYSIO',
                        'VALUE': value, 'VARIABLE': 'EEN_1'}, ignore_index=True)
        # EEN_2
        value = agg[[1]].mean()
        df = df.append({'ID_POINT': id_point, 'SOURCE': 'ERA5', 'TYPE': 'PHYSIO',
                        'VALUE': value, 'VARIABLE': 'EEN_2'}, ignore_index=True)
        # EEN_3
        value = agg[[ 2]].mean()
        df = df.append({'ID_POINT': id_point, 'SOURCE': 'ERA5', 'TYPE': 'PHYSIO',
                        'VALUE': value, 'VARIABLE': 'EEN_3'}, ignore_index=True)
        # EEN_4
        value = agg[[3]].mean()
        df = df.append({'ID_POINT': id_point, 'SOURCE': 'ERA5', 'TYPE': 'PHYSIO',
                        'VALUE': value, 'VARIABLE': 'EEN_4'}, ignore_index=True)


    df = df.sort_values(by=['ID_POINT'])
    df.to_sql('DON_AUX', con=conn, if_exists='replace', index=False)
    conn.close()


def create_physio_from_polygons(shp_path, id_name, nb_proc=1):
    """

    :param SHP_PATH:
    :param DB_PATH:
    :param OCC_SOL_100m_PATH:
    :param PENTE_250m_PATH:
    :param ALT_250m_PATH:
    :param options:
    :return:
    """

    OCC_SOL_100m_PATH = 'app/static/rasters/Physio/MODIS_100m_OCCUPATIONS_8C_.tif'
    print(OCC_SOL_100m_PATH)
    PENTE_250m_PATH = 'app/static/rasters/Physio/PENTE_deg_83.tif'
    ALT_250m_PATH = 'app/static/rasters/Physio/GMTED_DEM_250m.tif'
    #
    # OCC_SOL_100m_PATH = PROJECT_PATH + 'Physio/MODIS_100m_OCCUPATIONS_8C_.tif'
    # PENTE_250m_PATH = PROJECT_PATH + 'Physio/PENTE_deg_83.tif'
    # ALT_250m_PATH = PROJECT_PATH + 'Physio/GMTED_DEM_250m.tif'
    options = {1: 'PCON', 2: 'PFEU', 3: 'PMIX', 4: 'PLAC',
               5: 'PHUM', 6: 'PHER', 7: 'PBRU', 8: 'PROC'}

    df_occ = create_occ_sol_from_raster(shp_path=shp_path, raster_path=OCC_SOL_100m_PATH,
                                        var_physio_list=options, type_requete="OCC_SOL",
                                        source="HQP_100m_MODIS", var_name="",
                                        nb_processes=nb_proc, id_name=id_name)

    df_pente = create_occ_sol_from_raster(shp_path=shp_path, raster_path=PENTE_250m_PATH,
                                        var_physio_list="", type_requete="",
                                        source="GTDEM_250m", var_name="PTED",
                                        nb_processes=nb_proc, id_name=id_name)

    df_alt = create_occ_sol_from_raster(shp_path=shp_path, raster_path=ALT_250m_PATH,
                                        var_physio_list="", type_requete="",
                                        source="GTDEM_250m", var_name="ALTM",
                                        nb_processes=nb_proc, id_name=id_name)

    df_terr = create_occ_sol_from_raster(shp_path=shp_path, raster_path="",
                                         var_physio_list="", type_requete="TERRAIN",
                                         source="GTDEM_250m", var_name="",
                                         nb_processes=nb_proc, id_name=id_name)

    return pd.concat([df_occ, df_pente, df_alt, df_terr])


if __name__ == '__main__':
    print("main")