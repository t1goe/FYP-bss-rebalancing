import math
import os
import time
from os.path import isfile, join

import data_preprocessing

import pandas as pd
import matplotlib.pyplot as plt
import data_preprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm

from calendar import timegm


def station_get_weather(x):
    time_sec = int(time.mktime(time.strptime("2018-08-01 12", "%Y-%m-%d %H")))

    print(time_sec)


def get_weather_at_time(t, data):
    x = data[data['epoch'] == t]
    return x


def station_epoch_time(x):
    temp_x = x[:-6]
    return int(time.mktime(time.strptime(temp_x, "%Y-%m-%d %H")))


def attach_weather_data(path, station_file):
    if not os.path.exists(path + station_file):
        print("File: " + str(path + station_file) + " does not exist")
        return

    station_dataframe = pd.read_csv(path + station_file)

    if 'TIME' not in station_dataframe.columns:
        print('TIME column not present')
        return

    dest = './datasets/bss/dublin/reorg_plus_weather/'

    if not os.path.exists(dest):
        os.makedirs(dest)

    if os.path.exists(dest + station_file):
        # print("File plus weather already exists")
        return

    weather = pd.read_csv('datasets/weather/hly175/hly175clean.csv')

    new_weather_dataframe = pd.DataFrame(np.repeat(weather.values, 12, axis=0), columns=weather.columns)

    result = pd.concat([station_dataframe, new_weather_dataframe], axis=1)
    result.to_csv('datasets/bss/dublin/reorg_plus_weather/' + station_file, index=False)


# attach_weather_data('datasets/bss/dublin/reorg/station_2.csv')
# mypath = "./datasets/bss/dublin/reorg/"
#
# files = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
#
# for file in tqdm(files):
#     # print(file)
#     attach_weather_data(mypath, file)

data_preprocessing.organise_by_station()
# dataset = pd.read_csv('datasets/bss/dublin/reorg/station_2.csv', usecols=['TIME', 'AVAILABLE BIKES'])
# dataset.columns = ['TIME', 'BIKES']

# dataset['bikes(t-1)'] = dataset['bikes'].shift(1)

# attach_weather_data(dataset)

# utc_time = time.mktime(time.strptime("08-aug-2018 03:00", "%d-%b-%Y %H:%M"))

# print(utc_time)

# print(dataset)
