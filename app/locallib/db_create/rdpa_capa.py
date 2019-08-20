import requests
import schedule
import datetime
from pathlib import Path
import numpy as np


def job():
    print('##########################################################')
    print('[' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '] : NEW JOB STARTED: retrieving CAPA file')
    print('##########################################################')

    ORIGINAL_PATH = 'http://dd.weatheroffice.gc.ca/analysis/precip/rdpa/grib2/polar_stereographic/24/'
    DEST_PATH = ''

    for i in np.arange(0, 30, 1):
        date = (datetime.date.today() - datetime.timedelta(days=int(i))).strftime("%Y%m%d")
        filename = 'CMC_RDPA_APCP-024-0700cutoff_SFC_0_ps10km_' + date + '06_000.grib2'
        my_file = Path(DEST_PATH + filename)
        if my_file.exists() is False:
            print(i)
            rq = requests.get(ORIGINAL_PATH + filename)  # create HTTP response object
            if rq.status_code == 200:
                with open(DEST_PATH + filename, 'wb') as f:
                    f.write(rq.content)


schedule.every().day.at("16:30").do(job)
