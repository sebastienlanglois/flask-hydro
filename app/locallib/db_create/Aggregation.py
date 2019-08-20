import pandas as pd
import sqlite3


def hourly_to_daily_ERA5(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    sql = """   
                SELECT *
                FROM META_TS
                WHERE PAS_DE_TEMPS = "1_H"
                AND SOURCE = "ERA5"
                ORDER BY ID_SERIE
          """
    df_sup1 = pd.read_sql_query(sql, conn)
    conn.close()

    for idx, serie in df_sup1.iterrows():

        conn = sqlite3.connect(DB_PATH)

        sql = """ SELECT MAX(ID_SERIE) FROM META_TS"""
        #sql = """ SELECT MAX(ID_SERIE) FROM META_TS WHERE ID_SERIE NOT IN (SELECT MAX(ID_SERIE) FROM META_TS);"""
        max_value_id_serie = pd.read_sql_query(sql, conn).values[0][0]

        print(idx)
        index = serie['ID_SERIE']
        sql = """
                SELECT *
                FROM DON_TS
                WHERE ID_SERIE = "%s"
                ORDER BY DATE
              """% (index)

        df = pd.read_sql_query(sql, conn, index_col="DATE")
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_convert("America/Montreal")
        print(max_value_id_serie)
        df['ID_SERIE'] = int(max_value_id_serie + 1)


        serie['PAS_DE_TEMPS'] = '1_J'
        serie['DATE_DEBUT'] = '1978-12-31 00:00:00-05:00'
        serie['DATE_FIN'] = '2018-02-28 00:00:00-05:00'
        meta_ts = serie.copy().to_frame().T
        meta_ts['ID_SERIE'] = int(max_value_id_serie + 1)

        if serie['TYPE_SERIE'] == 'Temperature':
            df_max = df.resample('D').max()
            df_max['ID_SERIE'] = int(max_value_id_serie + 2)
            df_min = df.resample('D').min()
            df_min['ID_SERIE'] = int(max_value_id_serie + 3)
            df = df.resample('D').mean()
            df = pd.concat([df, df_max, df_min])

            df_smax = serie.copy()
            df_smax['TYPE_SERIE'] = 'Tmax'
            df_smax['ID_SERIE'] = int(max_value_id_serie + 2)

            df_smin = serie.copy()
            df_smin['TYPE_SERIE'] = 'Tmin'
            df_smin['ID_SERIE'] = int(max_value_id_serie + 3)
            meta_ts = pd.concat([meta_ts, df_smax.to_frame().T, df_smin.to_frame().T])
        elif serie['TYPE_SERIE'] in ["Precipitation", "Pluie", "Neige"]:
            df = df.resample('D').agg({'VALUE': 'sum', 'ID_SERIE': 'last'})
            meta_ts['AGGREGATION'] = "somme"
        else:
            df = df.resample('D').mean()

        df.reset_index(level=[0], inplace=True)
        df.round(3).to_sql('DON_TS', con=conn, if_exists='append', index=False)
        print(meta_ts)
        print(type(meta_ts['ID_SERIE']))

        meta_ts.to_sql('META_TS', con=conn, if_exists='append', index=False)
        conn.commit()

        print("Close sqlite file...")
        conn.close()


def daily_to_monthly(DB_PATH, aggreg_type):
    conn = sqlite3.connect(DB_PATH)
    sql = """   
                SELECT *
                FROM META_TS
                WHERE PAS_DE_TEMPS = "1_J"
                ORDER BY ID_SERIE
          """
    df_sup1 = pd.read_sql_query(sql, conn)
    conn.close()

    for idx, serie in df_sup1.iterrows():
        conn = sqlite3.connect(DB_PATH)

        sql = """ SELECT MAX(ID_SERIE) FROM META_TS"""
        #sql = """ SELECT MAX(ID_SERIE) FROM META_TS WHERE ID_SERIE NOT IN (SELECT MAX(ID_SERIE) FROM META_TS);"""
        max_value_id_serie = pd.read_sql_query(sql, conn).values[0][0]

        print(idx)
        index = serie['ID_SERIE']
        sql = """
                SELECT *
                FROM DON_TS
                WHERE ID_SERIE = "%s"
                ORDER BY DATE
              """% (index)

        df = pd.read_sql_query(sql, conn, index_col="DATE")
        df.index = pd.to_datetime(df.index, utc=True)
        df['ID_SERIE'] = int(max_value_id_serie + 1)


        serie['PAS_DE_TEMPS'] = '1_M'
        # serie['DATE_DEBUT'] = '1978-12-01 00:00:00-05:00'
        # serie['DATE_FIN'] = '2018-02-01 00:00:00-05:00'
        meta_ts = serie.copy().to_frame().T
        meta_ts['ID_SERIE'] = int(max_value_id_serie + 1)


        if serie['TYPE_SERIE'] in ["Precipitation", "Pluie", "Neige"]:
            df = df.resample('M').agg({'VALUE': 'sum', 'ID_SERIE': 'last'})
            meta_ts['AGGREGATION'] = "somme"
        else:
            if aggreg_type is "mean":
                df = df.resample('M').agg({'VALUE': 'mean', 'ID_SERIE': 'last'})
            elif aggreg_type is "max":
                df = df.resample('M').agg({'VALUE': 'max', 'ID_SERIE': 'last'})
                meta_ts['AGGREGATION'] = "max"
            elif aggreg_type is "min":
                df = df.resample('M').agg({'VALUE': 'min', 'ID_SERIE': 'last'})
                meta_ts['AGGREGATION'] = "min"

        df.reset_index(level=[0], inplace=True)

        if serie['TYPE_SERIE'] in ["Precipitation", "Pluie", "Neige"] and not aggreg_type == "mean":
            print("time series already in database...")
        else:
            print(max_value_id_serie)
            print(meta_ts)
            df.to_sql('DON_TS', con=conn, if_exists='append', index=False)
            meta_ts.to_sql('META_TS', con=conn, if_exists='append', index=False)
            conn.commit()
            print("Close sqlite file...")
        conn.close()

def daily_to_yearly(DB_PATH, aggreg_type):
    conn = sqlite3.connect(DB_PATH)
    sql = """   
                SELECT *
                FROM META_TS
                WHERE PAS_DE_TEMPS = "1_J"
                ORDER BY ID_SERIE
          """
    df_sup1 = pd.read_sql_query(sql, conn)
    conn.close()

    for idx, serie in df_sup1.iterrows():
        conn = sqlite3.connect(DB_PATH)

        sql = """ SELECT MAX(ID_SERIE) FROM META_TS"""
        #sql = """ SELECT MAX(ID_SERIE) FROM META_TS WHERE ID_SERIE NOT IN (SELECT MAX(ID_SERIE) FROM META_TS);"""
        max_value_id_serie = pd.read_sql_query(sql, conn).values[0][0]

        print(idx)
        index = serie['ID_SERIE']
        sql = """
                SELECT *
                FROM DON_TS
                WHERE ID_SERIE = "%s"
                ORDER BY DATE
              """% (index)

        df = pd.read_sql_query(sql, conn, index_col="DATE")
        df.index = pd.to_datetime(df.index, utc=True)
        df['ID_SERIE'] = int(max_value_id_serie + 1)


        serie['PAS_DE_TEMPS'] = '1_A'
        # serie['DATE_DEBUT'] = '1978-12-01 00:00:00-05:00'
        # serie['DATE_FIN'] = '2018-02-01 00:00:00-05:00'
        meta_ts = serie.copy().to_frame().T
        meta_ts['ID_SERIE'] = int(max_value_id_serie + 1)


        if serie['TYPE_SERIE'] in ["Precipitation", "Pluie", "Neige"]:
            df = df.resample('M').agg({'VALUE': 'sum', 'ID_SERIE': 'last'})
            meta_ts['AGGREGATION'] = "somme"
        else:
            if aggreg_type is "mean":
                df = df.resample('Y').agg({'VALUE': 'mean', 'ID_SERIE': 'last'})
            elif aggreg_type is "max":
                df = df.resample('Y').agg({'VALUE': 'max', 'ID_SERIE': 'last'})
                meta_ts['AGGREGATION'] = "max"
            elif aggreg_type is "min":
                df = df.resample('Y').agg({'VALUE': 'min', 'ID_SERIE': 'last'})
                meta_ts['AGGREGATION'] = "min"


        df.reset_index(level=[0], inplace=True)


        if serie['TYPE_SERIE'] in ["Precipitation", "Pluie", "Neige"] and not aggreg_type == "mean":
            print("time series already in database...")
        else:
            print(max_value_id_serie)
            print(meta_ts)
            df.to_sql('DON_TS', con=conn, if_exists='append', index=False)
            meta_ts.to_sql('META_TS', con=conn, if_exists='append', index=False)
            conn.commit()
            print("Close sqlite file...")
        conn.close()