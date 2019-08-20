import requests
import pandas as pd
import geopandas as gpd
import glob
import re
import numpy as np
import os
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
from multiprocessing import Pool
import sqlite3
import schedule
import time
import datetime
import pytz


# Functions
def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)


def function_requests(nom_fichier):

    print(nom_fichier)

    CEHQ_URL = "https://www.cehq.gouv.qc.ca/depot/historique_donnees/fichier/"

    rq = requests.get(CEHQ_URL + os.path.basename(nom_fichier[0].strip()) + '_Q.txt')  # create HTTP response object
    if rq.status_code == 200:
        with open(nom_fichier[0].strip() + '_Q.txt', 'wb') as f:
            f.write(rq.content)

    rn = requests.get(CEHQ_URL + os.path.basename(nom_fichier[0].strip()) + '_N.txt')  # create HTTP response object
    if rn.status_code == 200:
        with open(nom_fichier[0].strip() + '_N.txt', 'wb') as f:
            f.write(rn.content)



def CEHQ(f, DATA_PATH, DB_PATH, shp_path):
    print('##########################################################')
    print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] : NEW JOB STARTED: CEHQ update')
    print('##########################################################')

    ORIGINAL_PATH = 'https://www.cehq.gouv.qc.ca/hydrometrie/historique_donnees/ListeStation.asp?regionhydro=$&Tri=Non'

    nb_region = ["%02d" % n for n in range(0, 13)]
    regions = []
    print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] Getting all available stations...')
    for reg in nb_region:
        path = ORIGINAL_PATH.replace('$', reg)
        raw_html = simple_get(path)
        html = BeautifulSoup(raw_html, 'html.parser')

        for li in (html.select('area')):
            if li['href'].find('NoStation')>0:
                a = li['title'].split('-',1)
                a[0] = a[0].strip()
                a[0] = DATA_PATH + '/' + a[0]
                regions.append(a)
    print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] '+ str(len(regions)) + ' available stations...')
    # URL Request
    # Si parallèle
    # with Pool(8) as p:
    #     p.map(f, regions)
    # Sinon :
    # for nom_fichier in regions:
    #     f(nom_fichier, DATA_PATH)

    print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] Getting all available stations...done')

    print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] Parsing all available stations')
    # Parsing
    listeStations = glob.glob(DATA_PATH + "/*.txt")

    # Validate or correct file



    def del_if_2_cols(lines):
        if (len(lines[0].split()))!=4:
            lines.pop(0)
            del_if_2_cols(lines)
        return lines

    if os.name == 'nt':
        encoding = "ISO-8859-1"
    else:
        encoding = "ISO-8859-1"

    for file in listeStations:
        with open(file, 'r', encoding=encoding) as f:
            head = f.readlines()[0:21]
            f.close()
        with open(file, 'r', encoding=encoding) as f:
            lines = f.readlines()[22:]
            f.close()
        if lines and (len(lines[0].split())) != 4:
            print(os.path.splitext(os.path.basename(file))[0])
            del_if_2_cols(lines)
            text_fin = head + lines
            with open(file, "w", encoding=encoding)as f:
                f.write(''.join(map(str, text_fin)))
                f.close()

    list_coords = []
    list_sup = []
    list_stations = []
    list_type = []
    list_id = []
    liste_nom = []
    liste_regime = []

    liste_regions = np.array([os.path.basename(file) for file in np.array(regions)[:, 0]])

    # Ne retient que les stations pour lesquels un fichier de forme est disponible.
    # Plus contraignant, mais plus propre pour conserver une base de données robuste

    basename_listeStations = [ os.path.basename(x).split('.')[0].split("_")[0] for x in listeStations]

    print(basename_listeStations)
    print(len(basename_listeStations))

    lst_shp = sorted(glob.glob(shp_path + '/*/*.shp'))

    basename_lst_shp = [os.path.basename(x).split('.')[0].split("_")[0] for x in lst_shp]
    basename_lst_shp = ([*{*basename_lst_shp}])

    print(lst_shp)
    print(len(basename_lst_shp))


    available_stations = sorted(list(set(basename_listeStations).intersection(basename_lst_shp)))
    print(available_stations)
    print(len(available_stations))

    listOfIndices_unique = [basename_listeStations.index(key) for key in available_stations]
    print(listOfIndices_unique)

    listOfIndices_unique = [basename_listeStations.index(key) for key in available_stations]
    print(listOfIndices_unique)
    print(len(listOfIndices_unique))

    listOfIndices_not_unique = [i for y in available_stations for i, x in enumerate(basename_listeStations) if x == y]
    print(listOfIndices_not_unique)
    print(len(listOfIndices_not_unique))

    for file in [listeStations[i] for i in listOfIndices_not_unique]:
        with open(file, encoding=encoding) as f:
            lines = f.readlines()
        type_var = os.path.basename(file).replace('.', '_').split('_')[1]
        if type_var == 'Q':
            list_type.append('Debit')
        else:
            list_type.append('Niveau')
        stations = lines[2]
        itemindex = np.where(liste_regions == stations.split()[1])
        nom_long = regions[itemindex[0][0]][-1].strip()
        liste_nom.append(nom_long)
        type_var = os.path.basename(file).replace('.', '_').split('_')[1]
        liste_regime.append(lines[3][15:].split()[-1])
        list_stations.append(stations.split()[1])
        list_id.append(stations.split()[1] + '_' + type_var)
        superficie = float(lines[3][15:].split()[0])
        coords = lines[4][22:-2].split()
        list_sup.append(superficie)
        if len(coords) < 5:
            coords = [x for x in coords if x]
            list_coords.append([float(coords[0]), float(coords[2])])
        elif len(coords) >= 5:
            coords = [re.sub(r'[^-\d]', '', coord) for coord in coords]
            coords = [x for x in coords if x]
            list_coords.append([float(coords[0]) + float(coords[1]) / 60 + float(coords[2]) / 3600,
                                float(coords[3]) - float(coords[4]) / 60 - float(coords[5]) / 3600])
        else:
            print(file)
    print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] Parsing all available stations...done')

    print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] Making dataframes...')

    df_sup1 = pd.DataFrame(np.array([list_id, list_stations, liste_nom, list_type, liste_regime,list_coords, list_sup]).T,
                           columns=['STATION_ID', 'NUMERO_STATION', 'NOM_STATION',
                                    'TYPE_SERIE', 'REGIME', 'COORDS', 'SUPERFICIE'])


    # Importation de tous les fichiers .txt vers une liste de dataframe
    fields = ['DATE','VALUE']



    dict_df = {os.path.splitext(os.path.basename(station))[0] :
               pd.read_csv(station, skiprows=22,delim_whitespace=True,
                           usecols=[1, 2], names=fields, header=None,
                           encoding='latin1').fillna(np.nan)
               for station in [listeStations[i] for i in listOfIndices_not_unique]}


    key_to_delete = []
    for key, value in dict_df.items():
        value['VALUE'] = value['VALUE'].map(lambda x: float(str(x).lstrip('+-').rstrip('aAbBcC')))
        value = value.set_index('DATE')
        value.index = pd.to_datetime(value.index)
        value.index = value.index.tz_localize("Etc/GMT+5", ambiguous='infer', nonexistent='shift_forward')
        value.index = value.index.tz_convert("America/Montreal")
        value['STATION_ID'] = key
        dict_df[key] = value
        if len(value['VALUE'].dropna()) > 0:
            debut = value['VALUE'].dropna().index[0]
            fin = value['VALUE'].dropna().index[-1]
            df_sup1.loc[df_sup1.index[df_sup1['STATION_ID'] == key], 'DATE_DEBUT'] = debut
            df_sup1.loc[df_sup1.index[df_sup1['STATION_ID'] == key], 'DATE_FIN'] = fin
        else:
            df_sup1.drop(df_sup1.index[df_sup1['STATION_ID'] == key], inplace=True)
            key_to_delete.append(key)

    print(key_to_delete)
    for k in key_to_delete:
        dict_df.pop(k, None)
    df = pd.concat(dict_df.values())
    df.reset_index(level=0, inplace=True)

    # Index dataframe
    df_sup1 = df_sup1.sort_values(by=['NUMERO_STATION']).reset_index().drop(['index'], axis=1)
    df_sup1['LATITUDE'], df_sup1['LONGITUDE'] = df_sup1['COORDS'].map(lambda x: ':'.join(list(map(str, x)))).str.split(':', 1).str
    df_sup1 = df_sup1.drop('COORDS', axis=1)


    meta_sta_hydro = df_sup1.drop(columns=['STATION_ID','TYPE_SERIE', 'DATE_DEBUT', 'DATE_FIN'])
    meta_sta_hydro = meta_sta_hydro.drop_duplicates()
    meta_sta_hydro.insert(loc=2, column='PROVINCE', value='QC')
    meta_sta_hydro.insert(loc=0, column='ID_POINT', value=range(1000, 1000 + meta_sta_hydro.shape[0], 1))


    gdf_json = [gpd.read_file(shp).to_json() for shp in [shp_path + '/' + s + '/' + s + '.shp'
                                                         for s in available_stations]]
    meta_sta_hydro.insert(loc=8, column='GEOM', value=gdf_json)
    meta_sta_hydro.insert(loc=3, column='NOM_EQUIV', value=np.nan)

    meta_ts = df_sup1.drop(columns = ['NOM_STATION','REGIME','SUPERFICIE','LATITUDE','LONGITUDE'])
    meta_ts.insert(loc=3, column='PAS_DE_TEMPS', value='1_J')
    meta_ts.insert(loc=4, column='AGGREGATION', value='moy')
    meta_ts.insert(loc=5, column='UNITE', value='m3/s')
    meta_ts.insert(loc=8, column='SOURCE', value='CEHQ')

    meta_ts = pd.merge(meta_ts, meta_sta_hydro[['ID_POINT', 'NUMERO_STATION']],
                 left_on='NUMERO_STATION', right_on='NUMERO_STATION', how='left').drop(columns=['NUMERO_STATION'])

    cols = meta_ts.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    meta_ts = meta_ts[cols]


    meta_ts.insert(loc=0, column='ID_SERIE', value=range(1000, 1000 + meta_ts.shape[0], 1))

    df = pd.merge(df, meta_ts[['ID_SERIE', 'STATION_ID']],
                 left_on='STATION_ID', right_on='STATION_ID', how='left').drop(columns=['STATION_ID'])
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    meta_ts = meta_ts.drop(columns=['STATION_ID'])
    meta_ts['DATE_DEBUT'] = pd.to_datetime(meta_ts['DATE_DEBUT'])
    meta_ts['DATE_FIN'] = pd.to_datetime(meta_ts['DATE_FIN'])
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
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('DON_TS', con=conn, if_exists='replace', index = False)
    #df_montly.to_sql('CEHQ_MONTHLY_VALUES', con=conn,if_exists='replace')
    meta_sta_hydro.to_sql('META_STATION_BASSIN', con=conn,if_exists='replace', index=False)
    meta_ts.to_sql('META_TS', con=conn,if_exists='replace', index = False)
    cur = conn.cursor()
    sql = """    CREATE INDEX IF NOT EXISTS ID ON DON_TS(ID_SERIE);"""
    cur.execute(sql)
    conn.commit()
    conn.close()


#     #df_yearly.to_sql('CEHQ_YEARLY_VALUES', con=conn,if_exists='replace')
    print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] Dumping to SQL DB...done')
    print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] Job completed')
    print('  ')

if __name__ == '__main__':
    print("main")

