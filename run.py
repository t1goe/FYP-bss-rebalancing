import math
import os
import time

import pandas as pd
import matplotlib.pyplot as plt
import data_preprocessing
import pandas as pd
import numpy as np

from calendar import timegm


def attach_weather_data(x):
    if 'TIME' not in x.columns:
        print('TIME column not present')
        return

    weather = pd.read_csv('datasets/weather/hly175/hly175clean.csv')

    utc_time = time.strptime("08-aug-2018 03:00", "%d-%b-%Y %H:%M")
    print(utc_time)
    # epoch_time = timegm(utc_time)
    # -> 1236472051


dataset = pd.read_csv('datasets/bss/dublin/reorg/station_2.csv', usecols=['TIME', 'AVAILABLE BIKES'])
dataset.columns = ['TIME', 'BIKES']

# dataset['bikes(t-1)'] = dataset['bikes'].shift(1)

# attach_weather_data(dataset)

utc_time = time.strptime("08-aug-2018 03:00", "%d-%b-%Y %H:%M")
# utc_time.
print(utc_time)

# print(dataset)
