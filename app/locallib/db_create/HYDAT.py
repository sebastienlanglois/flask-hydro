import sqlite3
import pandas as pd
import datetime
import geopandas as gpd
import numpy as np
import glob
import glob
import os
import geopandas as gpd
import pytz
import datetime

shp_path = r'H:\Projets_communs\2019\SH-XX-XX-XX-HYD-CRU-FREQ-LaGrande\01_Intrants\06_Données_physio\shp'


def hydat_daily2(df, get_flow, to_plot):
    # Deux ou plus stations
    # station_number = "'02TC001' OR STATION_NUMBER='02TC002'"
    # station1 = "'02TC001'"
    # station2 = "'02TC002'"
    # qstr = "SELECT * FROM DLY_FLOWS WHERE STATION_NUMBER=%s OR STATION_NUMBER=%s\n" %(station1, station2)
    # qstr = SELECT * FROM DLY_FLOWS WHERE STATION_NUMBER='02TC001' OR STATION_NUMBER='02TC002'
    # ATTENTION PEUT ETRE TRES LONG

    # print(df)
    # if get_flow == True:
    #     header = "^FLOW\\d+"
    # else:
    #     header = "^LEVEL\\d+"
    header = "^FLOW\\d+" if get_flow == True else "^LEVEL\\d+"

    dly = df[["STATION_NUMBER", "YEAR", "MONTH"]]
    dly.columns = ["STATION_NUMBER", "YEAR", "MONTH"]
    dly
    # value.cols = df.columns[df.filter(regex="^FLOW\\d+")]
    # filter  sur les FLOW
    value = df.filter(regex=header)
    valuecols = value.columns
    # print(dlydata.shape)
    # now melt the data frame for data and flags
    dlydata = pd.melt(df, id_vars=["STATION_NUMBER", "YEAR", "MONTH"], value_vars=valuecols)

    if get_flow is True:
        dlydata["DAY"] = dlydata['variable'].apply(lambda x: np.int8(x[4:]))
    else:
        dlydata["DAY"] = dlydata['variable'].apply(lambda x: np.int8(x[5:]))
    # flowvariable = dlydata["variable"]
    # days = [x[4:6] for x in flowvariable]
    # dlydata["DAY"] = list(map(int, days))
    # censor ambiguous dates (e.g., 31st day for Apr, Jun, Sept, Nov)
    d = dlydata.loc[dlydata["MONTH"].isin([4, 6, 9, 11]) & (dlydata["DAY"] > 30)]
    d30 = d
    # print(d.index[:])
    # print(len(d))#

    if len(d) > 0:
        dlydata = dlydata.drop(d.index).reset_index(drop=True)
    # print(dlydata.shape)

    d = dlydata.loc[(dlydata["MONTH"].isin([2]) &
                     pd.to_datetime(dlydata["YEAR"], format='%Y').dt.is_leap_year &
                     (dlydata["DAY"] > 29))]
    if len(d) > 0:
        dlydata = dlydata.drop(d.index).reset_index(drop=True)
    d29 = d
    # print(dlydata.shape)

    d = dlydata.loc[(dlydata["MONTH"].isin([2]) &
                     ~pd.to_datetime(dlydata["YEAR"], format='%Y').dt.is_leap_year.values &
                     (dlydata["DAY"] > 28))]
    # print(d)
    if len(d) > 0:
        dlydata = dlydata.drop(d.index).reset_index(drop=True)
    d28 = d
    # print(dlydata.shape)
    # print(valuecols)

    # ----------------------------------SYMBOL--------------------------------------------------
    header_sym = "^FLOW_SYMBOL\\d+" if get_flow == True else "^LEVEL_SYMBOL\\d+"
    flag = df.filter(regex=header_sym)
    flagcols = flag.columns
    # print(flagcols)
    # ordonner les flag dans un dataframe
    dlyflags = pd.melt(df, id_vars=["STATION_NUMBER", "YEAR", "MONTH"], value_vars=flagcols)

    if len(d30) > 0:
        dlyflags = dlyflags.drop(d30.index).reset_index(drop=True)
    # print(dlyflags.shape)

    if len(d29) > 0:
        dlyflags = dlyflags.drop(d29.index).reset_index(drop=True)
    # print(dlyflags.shape)

    if len(d28) > 0:
        dlyflags = dlyflags.drop(d28.index).reset_index(drop=True)
    # print(dlyflags.shape)
    # -----------------------------------END SYMBOL---------------------------------------------

    # transform date
    dlydata.insert(loc=1, column='DATE', value=pd.to_datetime(dlydata[['YEAR', 'MONTH', 'DAY']]))
    # ---------------------------------plot the dataframe--------------------------------------
    dlytoplot = dlydata[['DATE', 'value']].set_index('DATE')
    dlydata = dlydata.drop(['YEAR', 'MONTH', 'DAY', 'variable'], axis=1)
    print(dlydata.shape)

    if to_plot == 1:
        dlytoplot.plot()
        return dlytoplot
    else:
        return dlydata


def HYDAT(shp_path, DB_PATH, source):
    cnx = sqlite3.connect(source)

    conn = sqlite3.connect(DB_PATH)

    df1 = pd.read_sql_query("SELECT * FROM STATIONS", cnx)
    df1 = df1[df1['PROV_TERR_STATE_LOC'].isin(['QC', 'ON', 'NB', 'NL'])]
    df1 = df1[['STATION_NUMBER', 'STATION_NAME', 'PROV_TERR_STATE_LOC', 'DRAINAGE_AREA_GROSS', 'LATITUDE', 'LONGITUDE']]
    df1.columns = ['NUMERO_STATION', 'NOM_STATION', 'PROVINCE', 'SUPERFICIE', 'LATITUDE', 'LONGITUDE']
    df1.insert(loc=0, column='STATION_ID', value=df1['NUMERO_STATION'])
    df1['STATION_ID'] = df1['NUMERO_STATION'].astype(str)
    df1.insert(loc=3, column='TYPE_SERIE', value='Debit')

    df2 = pd.read_sql_query("SELECT * FROM STN_REGULATION", cnx)
    df = pd.merge(df1, df2, how='left', left_on=['NUMERO_STATION'], right_on=['STATION_NUMBER'])
    df.insert(loc=5, column='REGIME', value=df['REGULATED'].map({0: 'Naturel', 1: 'Influencé', np.nan: 'Indisponible'}))
    df_sup1 = df.drop(columns=['YEAR_FROM', 'YEAR_TO', 'REGULATED', 'STATION_NUMBER'])

    meta_sta_hydro = df_sup1.drop(columns=['STATION_ID', 'TYPE_SERIE'])
    meta_sta_hydro = meta_sta_hydro.drop_duplicates()
    meta_sta_hydro.insert(loc=2, column='NOM_EQUIV', value=np.nan)
    meta_sta_hydro.insert(loc=0, column='ID_POINT', value=range(2000, 2000 + meta_sta_hydro.shape[0], 1))

    list_shp = glob.glob(shp_path + '/*/*.shp')
    basename_lst_shp = [os.path.basename(x).split('.')[0] for x in list_shp]
    basename_lst_shp = ([*{*basename_lst_shp}])

    available_stations1 = sorted(list(set(meta_sta_hydro['NUMERO_STATION'].values).intersection(basename_lst_shp)))


    ## equiv
    PROJECT_PATH = 'H:/BD_Base de donnees Hydro/02_Donnees/'
    DB_PATH2 = PROJECT_PATH + 'sqlite/HYDRO-dev.db'
    conn1 = sqlite3.connect(DB_PATH2)
    meta = pd.read_sql("SELECT NOM_EQUIV FROM META_STATION_BASSIN", conn1)
    available_stations2 = sorted(list(set(list(meta.values[meta.values != np.array(None)].flat)).symmetric_difference(available_stations1)))
    available_stations = sorted(
        list(set(available_stations2).intersection(basename_lst_shp)))
    conn1.close()

    gdf_json = [gpd.read_file(shp).to_json() for shp in [shp_path + '/' + s + '/' + s + '.shp'
                                                         for s in available_stations]]

    meta_sta_hydro = meta_sta_hydro[meta_sta_hydro['NUMERO_STATION'].isin(available_stations)]
    meta_sta_hydro.insert(loc=9, column='GEOM', value=gdf_json)

    meta_ts = df_sup1.drop(columns=['NOM_STATION', 'REGIME', 'SUPERFICIE', 'LATITUDE', 'LONGITUDE'])
    meta_ts = meta_ts[meta_ts['NUMERO_STATION'].isin(available_stations)]
    meta_ts.insert(loc=3, column='PAS_DE_TEMPS', value='1_J')
    meta_ts.insert(loc=4, column='AGGREGATION', value='moy')
    meta_ts.insert(loc=5, column='UNITE', value='m3/s')
    meta_ts.insert(loc=7, column='SOURCE', value='HYDAT')

    meta_ts = pd.merge(meta_ts, meta_sta_hydro[['ID_POINT', 'NUMERO_STATION']],
                       left_on='NUMERO_STATION', right_on='NUMERO_STATION', how='left').drop(columns=['NUMERO_STATION'])

    cols = meta_ts.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    meta_ts = meta_ts[cols]

    meta_ts.insert(loc=0, column='ID_SERIE', value=range(72871, 72871 + meta_ts.shape[0], 1))

    df = pd.merge(df, meta_ts[['ID_SERIE', 'STATION_ID']],
                  left_on='STATION_ID', right_on='STATION_ID', how='left').drop(columns=['STATION_ID'])
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    meta_ts = meta_ts.drop(columns=['STATION_ID'])

    meta_ts['DATE_DEBUT'] = np.nan
    meta_ts['DATE_FIN'] = np.nan

    meta_ts.to_sql('META_TS', con=conn, if_exists='append', index = False)
    meta_sta_hydro.to_sql('META_STATION_BASSIN', con=conn, if_exists='append', index = False)

    # Débit
    for idx, row in meta_sta_hydro.iterrows():
        numero_station = row['NUMERO_STATION']

        sql = """
           SELECT * 
           FROM DLY_FLOWS
           WHERE STATION_NUMBER in
           ("%s"
           )
           """ % (numero_station)
        chunk = pd.read_sql_query(sql, cnx)
        daily_station = hydat_daily2(chunk, True, False)
        daily_station.columns = ['ID_SERIE', 'DATE', 'VALUE']
        daily_station = daily_station.set_index(["DATE"])
        daily_station.index = pd.to_datetime(daily_station.index)
        daily_station.index = daily_station.index.tz_localize("America/Montreal", ambiguous='infer',
                                                              nonexistent='shift_forward')
        print(idx)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        daily_station['ID_SERIE'] = meta_ts[meta_ts['ID_POINT'] == row['ID_POINT']]['ID_SERIE'].values[0]
        daily_station.reset_index(level=0, inplace=True)
        daily_station.to_sql('DON_TS', con=conn, if_exists='append', index = False)
    #     da.to_sql('METADATA', con=conn,if_exists='append', index = False)

    # meta_ts['DATE_DEBUT'] = pd.to_datetime(meta_ts['DATE_DEBUT'])
    # meta_ts['DATE_FIN'] = pd.to_datetime(meta_ts['DATE_FIN'])

#
#     # # Monthly dataframe
#     # df_montly = df.groupby([df.index.year, df.index.month, 'STATION_ID']).mean()
#     # df_montly.columns = ['MONTHLY_VALUE']
#     # df_montly.index = df_montly.index.set_names(['YEAR', 'MONTH','STATION_ID'])
#     # df_montly.reset_index(inplace=True)
#     #
#     # # Yearly dataframe
#     # df_yearly = df_montly.groupby(['YEAR','STATION_ID']).mean().drop('MONTH',axis=1)
#     # df_yearly.columns = ['YEARLY_VALUE']
#     print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] Making dataframes...done')
#
#     print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] Dumping to SQL DB...')
#     # SQL Dump
#
#
    # conn = sqlite3.connect(DB_PATH)
    # df.to_sql('DON_TS', con=conn, if_exists='replace', index = False)
    # #df_montly.to_sql('CEHQ_MONTHLY_VALUES', con=conn,if_exists='replace')
    # meta_sta_hydro.to_sql('META_STATION_BASSIN', con=conn,if_exists='replace', index=False)
    # meta_ts.to_sql('META_TS', con=conn,if_exists='replace', index = False)
    # cur = conn.cursor()
    # sql = """    CREATE INDEX IF NOT EXISTS ID ON DON_TS(ID_SERIE);"""
    # cur.execute(sql)
    # conn.commit()
    # conn.close()


#     #df_yearly.to_sql('CEHQ_YEARLY_VALUES', con=conn,if_exists='replace')
    print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] Dumping to SQL DB...done')
    print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] Job completed')
    print('  ')

if __name__ == '__main__':
    print("main")
