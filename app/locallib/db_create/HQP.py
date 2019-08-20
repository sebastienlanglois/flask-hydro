import glob
import xarray as xr
from rasterio import features
from affine import Affine
import geopandas as gpd
from shapely import affinity
import sqlite3
import pytz
import numpy as np
import xarray as xr
import glob
import os
import pandas as pd
from lib.db_create.ERA5 import transform_from_latlon, rasterize
import multiprocessing as mp
import datetime
import pytz
import json

old_timezone = pytz.timezone("UTC")
new_timezone = pytz.timezone("America/Montreal")


class HQP_MET:

    def __init__(self, nc_file, polygon_file):
        self.nc_path = nc_file
        self.polygon_path = polygon_file
        with xr.open_rasterio(self.nc_path) as da:
            self.da = da
        self.dw = self.da
        self.gdf = polygon_file

    def add_shape_coord_from_data_array(self, coord_name):
        shp_gpd = self.gdf
        print(coord_name)
        shapes = [(shape, n) for n, shape in enumerate(self.gdf.geometry)]

        for shapes1, idx in shapes:
            if str(type(shapes1)) == "<class 'shapely.geometry.multipolygon.MultiPolygon'>":
                areas = [i.area for i in shapes1]
                # Get the area of the largest part
                max_area_index = areas.index(max(areas))
                shapes[idx] = (shapes1[max_area_index], idx)

        sw = 0
        for shapes1, idx in shapes:
            coords = self.da.coords
            longitude = 'x'
            latitude = 'y'

            transform = transform_from_latlon(coords[latitude], coords[longitude])
            out_shape = (len(coords[latitude]), len(coords[longitude]))
            raster = features.rasterize([shapes1], out_shape=out_shape,
                                        fill=np.nan, transform=transform,
                                        dtype=float)
            ct1 = 0


            print(idx)

            while np.any(~np.isnan(raster)) == False:
                if sw == 0:
                    print('--Correction pour les petits bassins--')
                    sw = 1
                ct1 += 1
                print('Bassin : ' + str(idx) + '-- itération ' + str(ct1))
                shapes2 = shapes[idx][0]
                shp_fct = affinity.scale(shapes2, xfact=1.5, yfact=1.5)
                raster = features.rasterize([shp_fct], out_shape=out_shape,
                                            fill=np.nan, transform=transform,
                                            dtype=float)

                shapes[idx] = (shp_fct, idx)
                print(np.nanmin(shapes[idx][0].exterior.coords.xy[0]))
                print(np.nanmax(shapes[idx][0].exterior.coords.xy[0]))

        self.da[coord_name] = rasterize(shapes, self.da.coords,
                                        longitude=longitude, latitude=latitude);
        return self

    def get_avg_var_BV(self, name):
        da = self.dw

        #         longitude_arr = (da['BV']==0).values.any(axis=0)*da.longitude.values
        #         longitude_arr[longitude_arr == 0] = np.nan

        latitude_arr = (da['BV']==0).values.any(axis=1)*da.latitude.values
        latitude_arr[latitude_arr == 0] = np.nan

        if np.any(~np.isnan(latitude_arr)):
            #             da = da[:,
            #                     (da.latitude>=np.nanmin(latitude_arr)) * (da.latitude<=np.nanmax(latitude_arr)) ,
            #                     (da.longitude>=np.nanmin(longitude_arr)) * (da.longitude<=np.nanmax(longitude_arr))]
            self.dw = da.where(da['BV'] == 0, drop=True)
        else:
            lon = round(self.gpd.geometry[0].centroid.coords.xy[0][0] / 0.25) * 0.25
            lat = round(self.gpd.geometry[0].centroid.coords.xy[1][0] / 0.25) * 0.25
            self.dw = self.da[:, self.da['latitude'] == lat, self.da['longitude'] == lon]
            print('1px')
        print(self.dw.coords)
        return self

    def aggregate_to_BV(self):
        self.dw = self.dw.mean(dim=['longitude', 'latitude'])
        return self

    def resample_time_sum(self, timestep):
        self.dw = self.dw.resample(time=timestep).sum(dim=xr.ALL_DIMS)
        return self

    def resample_time_mean(self, timestep):
        self.dw = self.dw.resample(time=timestep).mean()
        return self

    def reset_da(self):
        self.dw = self.da
        return self

    def reset_shp(self, polygon_file):
        self.gdf = polygon_file
        return self

    def get_all_BV_names(self):
        return self.gdf.NOM

    def clip_bv(self, name):
        bv_ids = {k: i for i, k in enumerate(self.gdf.NOM)}
        ds = self.da.where(self.da['BV'] == bv_ids[name], drop=True)
        return ds


def netcdf_to_sqlite(filespath, db_path, pool, id1, year, era, lst_shp, options, max_value_id_serie):

    list_da= []
    for variable in list(options.keys()):
        lst_asc = sorted(glob.glob(filespath + '/' + year + '/*/*/*' + variable + '.asc'))
        rasterlist = []
        for x in lst_asc:
            with xr.open_rasterio(x) as ds:
                name1 = os.path.basename(x)
                var = options[name1.split('.')[0].split('_')[-1]]
                heure = int(name1.split('.')[0].split('_')[-2])
                jour = int(name1.split('.')[0].split('_')[-3])
                mois = int(name1.split('.')[0].split('_')[-4])
                annee = int(name1.split('.')[0].split('_')[-5])
                ds['band'] = [datetime.datetime(annee, mois, jour, heure, 00)]
                rasterlist.append(ds)

        merged = xr.concat(rasterlist, dim='band').rename({"band": "time"}).rename(options[variable])
        list_da.append(merged)
    ds = xr.merge(list_da).load()

    print('Dataset conversion to dataframe...')
    list_df = pool.map(get_era_lat_long,
                       [(ds, id1, id2, name1, era, shp, max_value_id_serie)
                        for id2, shp in enumerate(lst_shp)])
    insert_ts_df_to_sqlite(pd.concat(list_df), db_path)


def get_era_lat_long(arg):
    ds, id1, id2, name1, era, meta_name, max_value_id_serie = arg
    print(str(id1) + ' - ' + name1 + ' : ' + str(id2))
    i, j = np.where(era.da[meta_name].values == 0)
    return process_ds_to_df(ds, i, j, id2, max_value_id_serie)


def insert_ts_df_to_sqlite(df, db_path):
    """
    """
    print("Dump into sqlite file...")
    conn = sqlite3.connect(db_path)
    df.to_sql('DON_TS', con=conn, if_exists='append', index=False)

    cur = conn.cursor()
    sql = """    CREATE INDEX IF NOT EXISTS ID ON DON_TS(ID_SERIE);
    """
    cur.execute(sql)
    conn.commit()
    print("Close sqlite file...")
    conn.close()
    return


# def correct_values_ERA5(x, y):
#     """
#     """
#     options = {'Neige': 'Neige', 'Pluie': 'Pluie',
#                'Precip': 'Precipitation', 'Tmax': 'Tmax',
#                'Tmin': 'Tmin', 'Tmoy': 'Temperature'}
#
#     z = x
#     if x in ['Neige', 'Pluie', 'Precip']:
#         y = y * 10
#         z = options[x]
#     else:
#         z = options[x]
#     return z,y


def process_ds_to_df(ds, i, j, id2, max_value_id_serie):
    """
    """
    list_df = []
    for idx, var in enumerate(ds.data_vars):
        num_vars = len(ds.data_vars)
        id_serie = max_value_id_serie + (id2*num_vars) + idx
        df1 = ds[var][:, i, j].mean(dim=['y', 'x']).to_dataframe(name='VALUE')
        df1.insert(0, 'ID_SERIE', id_serie)
        df1.insert(0, 'TYPE_SERIE', var)
        df1.index = df1.index.tz_localize("America/Montreal")
        df1.set_index('ID_SERIE', append=True, inplace=True)
        list_df.append(df1)

    df = pd.concat(list_df)

    if df.index[0] != 0:
        df.reset_index(level=[0, 1], inplace=True)

    # Correcting ERA5 values
    df.columns = ['DATE', 'ID_SERIE', 'TYPE_SERIE', 'VALUE']
    # df["TYPE_SERIE"], df["VALUE"] = zip(*df[['TYPE_SERIE', 'VALUE']].apply(lambda x:
    #                                                                        correct_values_ERA5(x.TYPE_SERIE,
    #                                                                                            x.VALUE,
    #                                                                                            ), axis=1))
    # # Correcting timezones
    # df['DATE'] = [old_timezone.localize(df['DATE'][i]).astimezone(new_timezone) for i in range(0, len(df['DATE']))]
    df = df.drop(columns=['TYPE_SERIE'])
    # df.insert(0, 'ID', df.apply(f, axis=1))

    return df


def correct_names_ERA5(x,y):
    """

    :param x:
    :param y:
    :return:
    """
    z = x
    if x in ['Neige', 'Pluie', 'Precipitation']:
        y = 'mm'
    elif x in ['Tmax', 'Tmin', 'Temperature']:
        y = 'C'
    return z, y


def create_weather_from_HQP(METEO_HQP_PATH, db_path, NB_PROC):

    pool = mp.Pool(processes=NB_PROC)

    options = {'Neige': 'Neige', 'Pluie': 'Pluie',
               'Precip': 'Precipitation', 'Tmax': 'Tmax',
               'Tmin': 'Tmin', 'Tmoy': 'Temperature'}

    variables = list(options.values())
    #lst_shp = sorted(glob.glob(shp_path + '/*/*.shp'))
    lst_asc = sorted(glob.glob(METEO_HQP_PATH + '/*/*/*/*.asc'))
    lst_asc_folder = sorted(glob.glob(METEO_HQP_PATH + '/*'))


    meta_ts = pd.DataFrame(columns=['ID_SERIE', 'ID_POINT', 'TYPE_SERIE', 'PAS_DE_TEMPS', 'AGGREGATION',
                                    'UNITE', 'DATE_DEBUT', 'DATE_FIN', 'SOURCE'])
    meta_ts = meta_ts.astype(dtype={"ID_SERIE": "int32", "ID_POINT": "int32",
                                    "TYPE_SERIE": "object", "PAS_DE_TEMPS": "object", "AGGREGATION": "object",
                                    "UNITE": "object", "DATE_DEBUT": "datetime64",
                                    "DATE_FIN": "datetime64", "SOURCE": "object"})

    conn = sqlite3.connect(db_path)
    sql = """ SELECT NUMERO_STATION, ID_POINT, GEOM FROM META_STATION_BASSIN"""
    meta_sta_hydro = pd.read_sql_query(sql, conn)
    sql = """ SELECT MAX(ID_SERIE) FROM META_TS"""
    max_value_id_serie = int(1 + pd.read_sql_query(sql, conn).values[0][0])
    conn.close()

    lst_shp = [gpd.GeoDataFrame.from_features(json.loads(poly)) for poly in meta_sta_hydro['GEOM']]
    era = HQP_MET(lst_asc[-1], lst_shp[1])


    # # Add all shapefiles infos to xarray
    for id1, shp in enumerate(lst_shp):
        era.reset_da()
        era.reset_shp(shp)
        name = meta_sta_hydro.loc[id1, "NUMERO_STATION"]
        era.add_shape_coord_from_data_array(name)

        for idx, var in enumerate(variables):
            num_vars = len(variables)
            id_point = meta_sta_hydro.loc[id1, "ID_POINT"]
            id_serie = max_value_id_serie + (id1 * num_vars) + idx

            row = [id_serie, id_point, var, '1_J', 'moy', 'mm',
                   '1960-01-01 00:00:00-05:00', '2018-12-31 00:00:00-05:00', 'Grilles VDM/Interpoleau']
            meta_ts.loc[len(meta_ts)] = row


    # # Loop on all years
    for id1, file in enumerate(lst_asc_folder):
        year = os.path.basename(file)
        netcdf_to_sqlite(METEO_HQP_PATH, db_path, pool, id1, year,
                         era, meta_sta_hydro["NUMERO_STATION"], options, max_value_id_serie)

    conn = sqlite3.connect(db_path)
    meta_ts["TYPE_SERIE"], meta_ts["UNITE"] = zip(*meta_ts[['TYPE_SERIE', 'UNITE']].
                                                  apply(lambda x:
                                                        correct_names_ERA5(x.TYPE_SERIE,
                                                                           x.UNITE,
                                                                           ), axis=1))
    meta_ts.drop_duplicates().to_sql('META_TS', con=conn, if_exists='append', index=False)
    conn.close()


def add_HQP_metadata(HQP_PATH, DB_PATH):
    """

    :param HQP_PATH:
    :param DB_PATH:
    :return:
    """
    lst_shp = sorted(glob.glob(HQP_PATH + '/*/*.shp'))
    df = pd.concat([gpd.read_file(shp) for shp in lst_shp])
    geom = [gpd.read_file(shp).to_json() for shp in lst_shp]

    meta_sta_hydro = df[["ID_POULPE", "Station", "SUPERFICIE", "geometry"]]
    meta_sta_hydro.insert(loc=1, column='NUMERO_STATION', value=meta_sta_hydro[["Station"]])
    meta_sta_hydro.insert(loc=3, column='NOM_EQUIV', value=np.nan)
    meta_sta_hydro.insert(loc=4, column='PROVINCE', value="QC")
    meta_sta_hydro.insert(loc=5, column='REGIME', value="Naturel")
    meta_sta_hydro.insert(loc=7, column='LATITUDE', value=np.nan)
    meta_sta_hydro.insert(loc=8, column='LONGITUDE', value=np.nan)
    meta_sta_hydro.rename(columns={'Station': 'NOM_STATION', 'ID_POULPE': 'ID_POINT',
                                   'geometry': 'GEOM'}, inplace=True)
    meta_sta_hydro["GEOM"] = geom

    conn = sqlite3.connect(DB_PATH)
    meta_sta_hydro.to_sql('META_STATION_BASSIN', con=conn, if_exists='append', index=False)
    conn.close()

def add_temp_metadata(HQP_PATH, DB_PATH):
    """

    :param HQP_PATH:
    :param DB_PATH:
    :return:
    """
    lst_shp = sorted(glob.glob(HQP_PATH + '/*/*.shp'))

    df = pd.concat([gpd.read_file(shp) for shp in lst_shp])
    geom = [gpd.read_file(shp).to_json() for shp in lst_shp]

    meta_sta_hydro = df[["OBJECTID_1", "NOM", "SUPERFICIE", "geometry"]]
    meta_sta_hydro.insert(loc=1, column='NUMERO_STATION', value=meta_sta_hydro[["NOM"]])
    meta_sta_hydro.insert(loc=3, column='NOM_EQUIV', value=np.nan)
    meta_sta_hydro.insert(loc=4, column='PROVINCE', value="QC")
    meta_sta_hydro.insert(loc=5, column='REGIME', value="Naturel")
    meta_sta_hydro.insert(loc=7, column='LATITUDE', value=np.nan)
    meta_sta_hydro.insert(loc=8, column='LONGITUDE', value=np.nan)
    meta_sta_hydro.rename(columns={'NOM': 'NOM_STATION', 'OBJECTID_1': 'ID_POINT',
                                   'geometry': 'GEOM'}, inplace=True)
    meta_sta_hydro["GEOM"] = geom

    conn = sqlite3.connect(DB_PATH)
    meta_sta_hydro.to_sql('META_STATION_BASSIN', con=conn, if_exists='append', index=False)
    conn.close()

if __name__ == '__main__':
    print("main")