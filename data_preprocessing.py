import os
import warnings
import zipfile
import wget
from pandas.errors import DtypeWarning
from tqdm import tqdm
import pandas as pd


def dublin_weather():
    """
    Download dublin weather data from met eireann
    """

    destination = './datasets/weather'
    url = "https://cli.fusio.net/cli/climate_data/webdata/hly175.zip"

    if not os.path.exists(destination):
        os.makedirs(destination)

    # Download if file does not exist
    # Downloads as zip, so code exists to unzip and remove it
    if not os.path.isfile(destination + "/hly175/hly175.csv"):
        print("Downloading hourly Dublin weather data")
        wget.download(url, destination)
        with zipfile.ZipFile(destination + "/hly175.zip", "r") as zip_ref:
            zip_ref.extractall(destination + "/hly175")
        os.remove(destination + "/hly175.zip")
    else:
        print("Dublin weather data already exists\n")


def dublin_bss():
    """
    Download dublin bss data with wget
    """
    destination = './datasets/bss/dublin'

    if not os.path.exists(destination):
        os.makedirs(destination)

    base_url = "https://data.smartdublin.ie/dataset/33ec9fe2-4957-4e9a-ab55-c5e917c7a9ab/resource"
    urls = ["/99a35442-6878-4c2d-8dff-ec43e91d21d7/download/dublinbikes_20200701_20201001.csv",
            "/8ddaeac6-4caf-4289-9835-cf588d0b69e5/download/dublinbikes_20200401_20200701.csv",
            "/aab12e7d-547f-463a-86b1-e22002884587/download/dublinbikes_20200101_20200401.csv",
            "/5d23332e-4f49-4c41-b6a0-bffb77b33d64/download/dublinbikes_20191001_20200101.csv",
            "/305d39ac-b6a0-4216-a535-0ae2ddf59819/download/dublinbikes_20190701_20191001.csv",
            "/76fdda3d-d8be-441b-92dd-0ee36d9c5316/download/dublinbikes_20190401_20190701.csv",
            "/538165d7-535e-4e1d-909a-1c1bfae901c5/download/dublinbikes_20190101_20190401.csv",
            "/67ea095f-67ad-47f5-b8f7-044743043848/download/dublinbikes_20181001_20190101.csv",
            "/9496fac5-e4d7-4ae9-a49a-217c7c4e83d9/download/dublinbikes_20180701_20181001.csv",
            "/2dec86ed-76ed-47a3-ae28-646db5c5b965/download/dublin.csv"]

    print("Downloading dublinbikes data")
    for url in tqdm(urls):
        filename = url.split("/")[-1]
        if filename in os.listdir(destination):
            # print("File \"" + filename + "\" already exists")
            continue

        final_url = base_url + url
        # print("starting download on " + filename)
        wget.download(final_url, destination)
        # print(filename + " : downloaded\n")

    print("Finished downloading Dublin BSS data")

    # Clear generated tmp files
    for file in os.listdir():
        if file[-4:] == ".tmp":
            os.remove(file)


def organise_by_station():
    """
    Reorganise all the dublinbikes CSVs by station instead of by quarter
    """
    # Get list of data files
    bss_files = os.listdir('./datasets/bss/dublin')
    if 'dublin.csv' in bss_files:
        bss_files.remove('dublin.csv')

    if 'reorg' in bss_files:
        bss_files.remove('reorg')

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

    print("Starting dublinbikes reorganisation")
    for station in station_ids:
        if os.path.exists('./datasets/bss/dublin/reorg/station_' + str(station) + '.csv'):
            # print('\tStation CSV already exists')
            continue

        # print('Working on station: ' + str(station) + loading_wheel[i], end='\r')
        print('Working on station: ' + str(station))
        df1 = pd.DataFrame(columns=columns)
        for file in bss_files:
            df2 = pd.read_csv('./datasets/bss/dublin/' + str(file))
            df2 = df2[df2['STATION ID'] == station]
            temp = [df1, df2]
            df1 = pd.concat(temp)
        df1 = df1.drop(df1.columns[[0]], axis=1)
        df1 = df1.drop(['LAST UPDATED', 'NAME', 'STATUS', 'ADDRESS', 'LATITUDE', 'LONGITUDE'], axis=1)

        df1.to_csv('./datasets/bss/dublin/reorg/station_' + str(station) + '.csv', index=False)

    print('\nFinished dublinbikes reorganisation')


def clean_weather_data():
    # Remove first n rows, that contain data from before the start of the BSS data
    with open('./datasets/weather/hly175/hly175.csv', 'r') as fin:
        data = fin.read().splitlines(True)
    with open('./datasets/weather/hly175/hly175clean.csv', 'w') as fout:
        fout.writelines(data[15])
        fout.writelines(data[131319:])

    # Remove all rows with non-useful data
    warnings.filterwarnings("ignore", category=DtypeWarning)
    dataset = pd.read_csv('./datasets/weather/hly175/hly175clean.csv',
                          usecols=['date', 'ind', 'rain', 'temp', 'rhum'])
    dataset.to_csv('./datasets/weather/hly175/hly175clean.csv', index=False)
