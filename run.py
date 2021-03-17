import math
import os
import time

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


def attach_weather_data(station_file):
    station_dataframe = pd.read_csv(station_file)

    if 'TIME' not in station_dataframe.columns:
        print('TIME column not present')
        return

    if not os.path.exists('datasets/bss/dublin/reorg_plus_weather'):
        os.makedirs('datasets/bss/dublin/reorg_plus_weather')

    weather = pd.read_csv('datasets/weather/hly175/hly175clean.csv')

    new_weather_dataframe = pd.DataFrame(np.repeat(weather.values, 12, axis=0), columns=weather.columns)

    result = pd.concat([station_dataframe, new_weather_dataframe], axis=1)
    result.to_csv(r'datasets/bss/dublin/reorg_plus_weather/station_2.csv', index=False)


attach_weather_data('datasets/bss/dublin/reorg/station_2.csv')

# data_preprocessing.clean_weather_data()

# dataset = pd.read_csv('datasets/bss/dublin/reorg/station_2.csv', usecols=['TIME', 'AVAILABLE BIKES'])
# dataset.columns = ['TIME', 'BIKES']

# dataset['bikes(t-1)'] = dataset['bikes'].shift(1)

# attach_weather_data(dataset)

# utc_time = time.mktime(time.strptime("08-aug-2018 03:00", "%d-%b-%Y %H:%M"))

# print(utc_time)

# print(dataset)
