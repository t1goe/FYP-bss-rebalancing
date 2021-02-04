import os
import pandas as pd
from tqdm import tqdm


# from numba import jit


def by_station():
    # Get list of data files
    bssfiles = os.listdir('./datasets/bss/dublin')
    if 'dublin.csv' in bssfiles:
        bssfiles.remove('dublin.csv')

    if 'reorg' in bssfiles:
        bssfiles.remove('reorg')

    # Get all the station IDs
    dataset = pd.read_csv('./datasets/bss/dublin/dublin.csv',
                          usecols=['Number'])
    station_ids = []
    for d in dataset['Number'].unique():
        station_ids.append(d)
    station_ids.sort()

    # Get column names
    columns = pd.read_csv('./datasets/bss/dublin/dublinbikes_20200701_20201001.csv', nrows=1).columns

    # Create the directory if it does not exist
    destination = './datasets/bss/dublin/reorg'
    if not os.path.exists(destination):
        os.makedirs(destination)

    for station in station_ids:
        print('\nWorking on station: ' + str(station))
        if os.path.exists('./datasets/bss/dublin/reorg/station_' + str(station) + '.csv'):
            print('\tStation CSV already exists')
            continue

        df1 = pd.DataFrame(columns=columns)
        for file in tqdm(bssfiles):
            df2 = pd.read_csv('./datasets/bss/dublin/' + str(file))
            df2 = df2[df2['STATION ID'] == station]
            temp = [df1, df2]
            df1 = pd.concat(temp)
        df1.to_csv('./datasets/bss/dublin/reorg/station_' + str(station) + '.csv')
