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
import json
import pandas as pd


old_timezone = pytz.timezone("UTC")
new_timezone = pytz.timezone("America/Montreal")


def transform_from_latlon(lat, lon):
    """ input 1D array of lat / lon and output an Affine transformation
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def rasterize(shapes, coords, latitude='latitude', longitude='longitude',
              fill=np.nan, **kwargs):
    """
    """
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))


def correct_values_ERA5(x, y, options):
    """
    """
    z = x
    if x in ['tp', 'sd', 'e']:
      y = y * 1000
      z = options[x]
    elif x in ['t2m']:
      y = y - 273.15
      z = options[x]
    elif x in ['u10', 'v10']:
      z = options[x]
    return z,y


def correct_names_ERA5(x,y, options):
    """
    """
    z = x
    if x in ['tp', 'sd', 'e']:
        y = 'mm'
        z = options[x]
    elif x in ['t2m']:
        y = 'C'
        z = options[x]
    elif x in ['u10', 'v10']:
        y = 'm/s'
        z = options[x]
    return z, y


def f(x):
    """
    """
    return int(str(int(pd.to_datetime(x[0]).timestamp())) + str(x[1]))


def insert_ts_df_to_sqlite(df, db_path, options):
    """
    """
    print("Dump into sqlite file...")
    conn = sqlite3.connect(db_path)
    #df.to_sql('RESERVED_TEMP', con=conn, if_exists='replace', index=True)
    df.to_sql('DON_TS', con=conn, if_exists='append', index=False)

    cur = conn.cursor()
    sql = """    CREATE INDEX IF NOT EXISTS ID ON DON_TS(ID_SERIE);
    """
    cur.execute(sql)
    conn.commit()

    print("Close sqlite file...")
    conn.close()
    return


def separation_pluie_neige(precip, temp):
    if temp is np.nan:
        precip_liquide = np.nan
        precip_solide  = np.nan
    elif temp > 2:
        precip_liquide = precip
        precip_solide = 0
    elif temp < -2:
        precip_liquide = 0
        precip_solide = precip
    else:
        coef = np.interp(temp, [-2, 2], [0, 1])
        precip_solide = (1-coef)*precip
        precip_liquide = coef*precip
    return precip_liquide, precip_solide


def process_ds_to_df(ds, i, j, id2, options, max_value_id_serie):
    """
    """
    list_df = []
    num_vars = len(ds.data_vars) + 2

    for idx, var in enumerate(ds.data_vars):
        id_serie = max_value_id_serie + (id2*num_vars) + idx
        df1 = ds[var][:, i, j].mean(dim=['latitude', 'longitude']).to_dataframe(name='VALUE')
        df1.insert(0, 'ID_SERIE', id_serie)
        df1.insert(0, 'TYPE_SERIE', var)
        df1.set_index('ID_SERIE', append=True, inplace=True)
        list_df.append(df1)

    df = pd.concat(list_df)
    if df.index[0] != 0:
        df.reset_index(level=[0, 1], inplace=True)

    # Correcting ERA5 values
    df.columns = ['DATE', 'ID_SERIE', 'TYPE_SERIE', 'VALUE']
    df["TYPE_SERIE"], df["VALUE"] = zip(*df[['TYPE_SERIE', 'VALUE']].apply(lambda x:
                                                                           correct_values_ERA5(x.TYPE_SERIE, x.VALUE,
                                                                                               options), axis=1))
    # Correcting timezones
    df['DATE'] = [old_timezone.localize(df['DATE'][i]).astimezone(new_timezone) for i in range(0, len(df['DATE']))]



    df_precip = df[df['TYPE_SERIE'].isin(['Precipitation', 'Temperature'])]
    df_precip = pd.pivot_table(df_precip, index="DATE", columns='TYPE_SERIE',
                               values='VALUE').rename_axis(None, axis=1)
    df_precip["Pluie"], \
    df_precip["Neige"] = zip(*df_precip[['Precipitation', 'Temperature']].
                             apply(lambda x: separation_pluie_neige(x['Precipitation'],
                                                                    x['Temperature']), axis=1))
    df2 = df_precip.drop(columns=['Precipitation', 'Temperature'])
    df3 = df2.stack()
    df_aj = pd.DataFrame(df3)
    df_aj.reset_index(level=[0, 1], inplace=True)
    df_aj.columns = ["DATE", "TYPE_SERIE", "VALUE"]

    df_aj.insert(1, 'ID_SERIE', 0)
    for idx, var in enumerate(["Pluie","Neige"]):
        id_serie = max_value_id_serie + (id2 * num_vars) + len(ds.data_vars) + idx
        df_aj.loc[df_aj["TYPE_SERIE"] == var, 'ID_SERIE'] = id_serie

    df = pd.concat([df, df_aj])

    df = df.drop(columns=['TYPE_SERIE'])
    return df.round(3)


def netcdf_to_sqlite(nc, db_path, variables, pool, id1, name1, era, options, lst_shp, max_value_id_serie):
    with xr.open_dataset(nc)[variables].load() as ds:
        list_df = pool.map(get_era_lat_long,
                           [(ds, id1, id2, name1, era, shp, options, max_value_id_serie)
                            for id2, shp in enumerate(lst_shp)])
        insert_ts_df_to_sqlite(pd.concat(list_df), db_path, options)


def netcdf_to_df(nc, variables, pool, id1, name1, era, options, lst_shp):
    with xr.open_dataset(nc)[variables].load() as ds:
        list_df = pool.map(get_era_lat_long,
                           [(ds, id1, id2, name1, era, shp, options, 1000)
                            for id2, shp in enumerate(lst_shp)])
        return pd.concat(list_df)


def get_era_lat_long(arg):
    ds, id1, id2, name1, era, meta_name, options, max_value_id_serie = arg
    print(str(id1) + ' - ' + name1 + ' : ' + str(id2))
    i, j = np.where(era.da[meta_name].values == 0)
    return process_ds_to_df(ds, i, j, id2, options, max_value_id_serie)


def create_weather_from_ERA5(nc_path, db_path, options, NB_PROC):
    import multiprocessing as mp
    pool = mp.Pool(processes=NB_PROC)

    #lst_shp = sorted(glob.glob(shp_path + '/*/*.shp'))
    #lst_shp.sort()
    lst_nc = sorted(glob.glob(nc_path + '/*.nc'))
    #lst_nc.sort()
    variables = list(options.keys())



    meta_ts = pd.DataFrame(columns=['ID_SERIE', 'ID_POINT', 'TYPE_SERIE', 'PAS_DE_TEMPS', 'AGGREGATION',
                                    'UNITE', 'DATE_DEBUT', 'DATE_FIN', 'SOURCE'])
    meta_ts = meta_ts.astype(dtype={"ID_SERIE": "int32", "ID_POINT": "int32",
                                    "TYPE_SERIE": "object", "PAS_DE_TEMPS": "object", "AGGREGATION": "object",
                                    "UNITE": "object", "DATE_DEBUT": "datetime64",
                                    "DATE_FIN": "datetime64", "SOURCE": "object"})
    # Add all shapefiles infos to xarray

    conn = sqlite3.connect(db_path)
    sql = """ SELECT NUMERO_STATION, ID_POINT, GEOM FROM META_STATION_BASSIN"""
    meta_sta_hydro = pd.read_sql_query(sql, conn)
    sql = """ SELECT MAX(ID_SERIE) FROM META_TS"""
    max_value_id_serie = int(1 + pd.read_sql_query(sql, conn).values[0][0])
    conn.close()

    lst_shp = [gpd.GeoDataFrame.from_features(json.loads(poly)) for poly in meta_sta_hydro['GEOM']]
    era = BV_ERA5(lst_nc[-1], lst_shp[1], 'tp')

    for id1, shp in enumerate(lst_shp):
        era.reset_da()
        era.reset_shp(shp)
        name = meta_sta_hydro.loc[id1, "NUMERO_STATION"]
        era.add_shape_coord_from_data_array(name)
        num_vars = len(variables) + 2


        for idx, var in enumerate(variables + ["Pluie", "Neige"]):
            id_point = meta_sta_hydro.loc[id1, "ID_POINT"]
            id_serie = max_value_id_serie + (id1 * num_vars) + idx
            row = [id_serie, id_point, var, '1_H', 'moy', 'mm',
                   '1978-12-31 19:00:00-05:00', '2018-12-31 18:00:00-05:00', 'ERA5']
            meta_ts.loc[len(meta_ts)] = row

    # # Loop on all ERA5 netcdfs
    for id1, nc in enumerate(lst_nc):
        name1 = os.path.basename(nc)
        netcdf_to_sqlite(nc, db_path, variables, pool, id1, name1,
                         era, options, meta_sta_hydro["NUMERO_STATION"], max_value_id_serie)

    conn = sqlite3.connect(db_path)
    meta_ts["TYPE_SERIE"], meta_ts["UNITE"] = zip(*meta_ts[['TYPE_SERIE', 'UNITE']].
                                                  apply(lambda x:
                                                        correct_names_ERA5(x.TYPE_SERIE,
                                                                           x.UNITE,
                                                                           options), axis=1))
    meta_ts.drop_duplicates().to_sql('META_TS', con=conn, if_exists='append', index=False)


def create_weather_from_ERA5_simple(shp_path, nc_path, options, NB_PROC):
    import multiprocessing as mp
    pool = mp.Pool(processes=NB_PROC)

    lst_shp = sorted(glob.glob(shp_path + '/*.shp'))
    lst_nc = sorted(glob.glob(nc_path + '/*.nc'))
    variables = list(options.keys())

    era = BV_ERA5(lst_nc[-1], gpd.read_file(lst_shp[1]), 'tp')

    meta_ts = pd.DataFrame(columns=['ID_SERIE', 'ID_POINT', 'TYPE_SERIE', 'PAS_DE_TEMPS', 'AGGREGATION',
                                    'UNITE', 'DATE_DEBUT', 'DATE_FIN', 'SOURCE'])
    meta_ts = meta_ts.astype(dtype={"ID_SERIE": "int32", "ID_POINT": "object",
                                    "TYPE_SERIE": "object", "PAS_DE_TEMPS": "object", "AGGREGATION": "object",
                                    "UNITE": "object", "DATE_DEBUT": "datetime64",
                                    "DATE_FIN": "datetime64", "SOURCE": "object"})

    # Add all shapefiles infos to xarray
    for id1, shp in enumerate(lst_shp):
        era.reset_da()
        era.reset_shp(gpd.read_file(shp))
        name = os.path.basename(shp).split('.')[0]
        print(name)
        era.add_shape_coord_from_data_array(name)
        num_vars = len(variables) + 2
        for idx, var in enumerate(variables + ["Pluie", "Neige"]):
            id_serie = 1000 + (id1 * num_vars) + idx
            row = [id_serie, name, var, '1_H', 'moy', 'mm',
                   '1978-12-31 19:00:00-05:00', '2019-02-28 18:00:00-05:00', 'ERA5']
            meta_ts.loc[len(meta_ts)] = row

    # # Loop on all ERA5 netcdfs
    list_df = []
    list_names = [os.path.basename(name).split('.')[0] for name in lst_shp]
    for id1, nc in enumerate(lst_nc):
        name1 = os.path.basename(nc)
        list_df.append(netcdf_to_df(nc, variables, pool, id1, name1, era, options, list_names))

    meta_ts["TYPE_SERIE"], meta_ts["UNITE"] = zip(*meta_ts[['TYPE_SERIE', 'UNITE']].
                                                  apply(lambda x:
                                                        correct_names_ERA5(x.TYPE_SERIE,
                                                                           x.UNITE,
                                                                           options), axis=1))

    return pd.concat(list_df), meta_ts


class BV_ERA5:

    def __init__(self, nc_file, polygon_file, variables):
        self.nc_path = nc_file
        self.polygon_path = polygon_file
        with xr.open_dataset(self.nc_path,  chunks={'time': 744,
                                                    'latitude': 81,
                                                    'longitude': 109})[variables] as da:
            self.da = da
        self.dw = self.da
        self.gdf = polygon_file

    def add_shape_coord_from_data_array(self, coord_name):
        shp_gpd = self.gdf

        shapes = [(shape, n) for n, shape in enumerate(self.gdf.geometry)];
        sw = 0
        for shapes1, idx in shapes:
            coords = self.da.coords
            longitude = 'longitude'
            latitude = 'latitude'

            transform = transform_from_latlon(coords[latitude], coords[longitude])
            out_shape = (len(coords[latitude]), len(coords[longitude]))
            raster = features.rasterize([shapes1], out_shape=out_shape,
                                        fill=np.nan, transform=transform,
                                        dtype=float)
            ct1 = 0
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
                                        longitude='longitude', latitude='latitude');
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

    def reset_shp(self, gdf):
        self.gdf = gdf
        return self

    def get_all_BV_names(self):
        return self.gdf.NOM

    def clip_bv(self, name):
        bv_ids = {k: i for i, k in enumerate(self.gdf.NOM)}
        ds = self.da.where(self.da['BV'] == bv_ids[name], drop=True)
        return ds