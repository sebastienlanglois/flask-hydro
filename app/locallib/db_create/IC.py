import sqlite3
import pandas as pd
import datetime
import geopandas as gpd
import numpy as np
import glob
import glob
import os
import pytz
import datetime
import cx_Oracle
from sqlalchemy import create_engine
import psycopg2


shp_path = r'H:\Projets_communs\2019\SH-XX-XX-XX-HYD-CRU-FREQ-LaGrande\01_Intrants\06_Données_physio\shp'


def make_meta_ts_table_from_IC(ts_ic, list_id_dest, current_id_serie, type_serie, unite):

    list_id_dest['nom_equiv'] = list_id_dest['nom_equiv'].astype('int')
    dates = ts_ic.reset_index().groupby('ID_DEST')['DATEHEURE'].agg(['first', 'last'])

    meta_ts = pd.merge(list_id_dest, dates, right_index=True, left_on='nom_equiv')
    meta_ts.columns = ['id_point', 'nom_equiv','date_debut', 'date_fin']
    meta_ts.insert(loc=0, column='id_serie',
                   value=range(current_id_serie + 1, current_id_serie + 1 + meta_ts.shape[0], 1))
    meta_ts.insert(loc=3, column='type_serie', value= type_serie)
    meta_ts.insert(loc=4, column='pas_de_temps', value='1_J')
    meta_ts.insert(loc=5, column='aggregation', value='moy')
    meta_ts.insert(loc=6, column='unite', value= unite)
    meta_ts.insert(loc=9, column='SOURCE', value='IC')
    return meta_ts


def sql_from_IC(var,value, engine, cnx, sqlite, cpt):
    #current_id_serie = pd.read_sql_query("SELECT MAX(ID_SERIE) FROM META_TS", engine).values[0][0]

    if cpt == 1:
        current_id_serie = 1
    else:
        current_id_serie = pd.read_sql_query("SELECT MAX(ID_SERIE) FROM META_TS", sqlite).values[0][0]

    list_id_dest = pd.read_sql_query("SELECT id_point, nom_equiv FROM bassins where id_point>10000", engine)

    type_serie, unite, table, pdt, value_agg, proc = value

    sql = """
    SELECT d.HIST_TIMESTAMP AS DATEHEURE, d.{} AS VALEUR, pd.ID_DESTINATION AS ID_DEST
    FROM ddex.con_pointdonnee pd
    join ddex.don_donneehistorique dh on pd.id_analog = dh.id_analog
        and dh.pastemps = {} -- Journalier
        and dh.processus in {} -- Hydrologie et obligations d'affaires, Prévision hydrologique (SPAN)
    join ddex.{} d on d.id_analog = pd.id_analog 
    join ddex.analog_key ak on pd.id_analog = ak.id_analog
    WHERE pd.typedonneeexploitation = 1 -- Historique
    and pd.genredestination = 1 -- ESP
        and pd.id_destination IN {} -- ESP
        and pd.typepointdonnee = {} -- type de point
        and d.HIST_TIMESTAMP > 
        TO_DATE('1960-01-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
    ORDER BY d.HIST_TIMESTAMP ASC
    """.format(value_agg, pdt, proc, table,  tuple(list_id_dest['nom_equiv'].values.astype(int)), int(var))

    ts_ic = pd.read_sql_query(sql, cnx, index_col='DATEHEURE')
    if var is '675':
        # Correction pour la données de neige
        ts_ic.index = (pd.to_datetime(ts_ic.index) + pd.DateOffset(-1)).tz_localize('Etc/GMT-12')
        ts_ic.index = ts_ic.index.tz_convert('America/Montreal')
    else:
        ts_ic.index = (pd.to_datetime(ts_ic.index) + pd.DateOffset(-1)).tz_localize('America/Montreal')

    return ts_ic, list_id_dest, current_id_serie


def IC(options, DB_PATH):
    cnx = cx_Oracle.connect('DB6282', 'hydDB6282', 'psse3zos1200-scan.hqp.hydro.qc.ca:1534/DPP_SNAP3')
    engine = create_engine('postgresql+psycopg2://postgres:postgres@10.4.151.163:5439/hydro_db', echo=False)
    sqlite = sqlite3.connect(DB_PATH)
    cpt = 0
    for key, value in options.items():
        cpt = cpt + 1
        type_serie, unite, table, pdt, value_agg, proc = value
        print(type_serie)
        print('querying databases')
        ts_ic, list_id_dest, current_id_serie = sql_from_IC(key, value,engine, cnx, sqlite, cpt)
        # meta data for time series (dataframe)
        print('convert output to dataframe')
        meta_ts = make_meta_ts_table_from_IC(ts_ic, list_id_dest, current_id_serie, type_serie, unite)

       # Time series dataframe
        df = pd.merge(meta_ts[['id_serie', 'nom_equiv']], ts_ic.reset_index(),
                      left_on='nom_equiv', right_on='ID_DEST', how='right').drop(columns=['nom_equiv', 'ID_DEST'])
        df.columns = ['id_serie', 'date', 'value']
        meta_ts = meta_ts.drop(columns='nom_equiv')
        # TO SQL

        meta_ts.to_sql('meta_ts', con=sqlite, if_exists='append', index=False)
        df.to_sql('don_ts', con=sqlite, if_exists='append', index=False)  # changer nom table
        print(meta_ts.head())
        print(df.head())

    cur = sqlite.cursor()
    sql = """    CREATE INDEX IF NOT EXISTS ID ON DON_TS(ID_SERIE, DATE);"""
    cur.execute(sql)
    sqlite.commit()
    sqlite.close()



#     #df_yearly.to_sql('CEHQ_YEARLY_VALUES', con=conn,if_exists='replace')
    print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] Dumping to SQL DB...done')
    print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] Job completed')
    print('  ')

if __name__ == '__main__':
    print("main")
