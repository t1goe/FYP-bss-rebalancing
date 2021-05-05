from datetime import datetime

import math
import pickle
from os import listdir
from os.path import isfile, join
import os
import csv

from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
import warnings
from tqdm import tqdm
import pandas

d = datetime(year=2019, month=12, day=3)


def get_station_names():
    mypath = '../datasets/bss/dublin/ml_models/'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    station_ids = [x.split('.')[0].split('_')[1] for x in files]

    output = {}

    for sid in station_ids:
        csv_file = csv.reader(open('../datasets/bss/dublin/original/dublin.csv', "r"), delimiter=",")
        for row in csv_file:
            if sid == row[0]:
                output[sid] = row[1]

    return output


def simple_predict(station_id, int_time, int_date, int_day):
    destination_directory = '../datasets/bss/dublin/simple_ml_models/'
    scaler_destination_directory = copy.deepcopy(destination_directory) + 'scalers/'

    model = tf.keras.models.load_model(destination_directory + 'station_' + str(station_id) + '.h5')

    file = open(scaler_destination_directory + 'station_' + str(station_id) + '.pkl', "rb")
    scaler = pickle.load(file)
    file.close()

    params = np.array([0, int_time, int_date, int_day])
    params = params.reshape(1, -1)
    params = scaler.transform(params)
    params = np.array([params])
    params = params.tolist()
    params[0][0].pop(0)
    params = np.array(params)

    answer = model.predict(params)
    full_row = concatenate((answer, params[0]), axis=1)
    inv_row = scaler.inverse_transform(full_row)

    return inv_row[0][0]


def get_station_capacities():
    df = pandas.read_csv('../datasets/bss/dublin/original/dublinbikes_20180701_20181001.csv',
                         usecols=['STATION ID', 'BIKE STANDS'])
    df = df.drop_duplicates()
    output = {}
    for index, row in df.iterrows():
        output[int(row['STATION ID'])] = int(row['BIKE STANDS'])

    return output


def generate_report(date=datetime(year=2019, month=8, day=1)):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    destination_directory = '../outputs/date_reports/'
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    if os.path.exists(destination_directory + date.strftime("%Y-%m-%d") + '.csv'):
        print("file already exists")
        return

    with open(destination_directory + date.strftime("%Y-%m-%d") + '.csv', 'w', newline='') as csvfile:
        times = ["station"]
        for x in range(24):
            times.append(str(x) + ":00")

        writer = csv.DictWriter(csvfile, fieldnames=times)

        writer.writeheader()

        stations = get_station_names()

        for sid, name in tqdm(stations.items()):
            answers = {"station": sid}
            for x in range(24):
                answers[str(x) + ":00"] = (
                    round(
                        simple_predict(
                            sid,
                            (x * 12),
                            date.strftime('%j'),
                            date.strftime('%w')
                        )
                    )
                )
            writer.writerow(answers)


# generate_report(d)


def get_overunder_population(pop_dict, optimal_capacity, upper_capacity, lower_capacity):
    # Get the maximum capacity of each station as dict
    station_caps = get_station_capacities()

    overpopulation_list = {}
    underpopulation_list = {}
    for station_id, pop in pop_dict.items():

        if pop < 0:
            continue

        try:
            if int(station_caps[station_id] * upper_capacity) < pop:
                overpopulation_list[station_id] = [
                    int(pop - (station_caps[station_id] * optimal_capacity)),
                    pop / station_caps[station_id],
                    pop
                ]

            if int(station_caps[station_id] * lower_capacity) > pop:
                underpopulation_list[station_id] = [
                    int((station_caps[station_id] * optimal_capacity) - pop),
                    pop / station_caps[station_id],
                    pop
                ]

        except KeyError as e:
            print(e)
            continue

    overpopulation_list = dict(sorted(overpopulation_list.items(), key=lambda item: item[1][1], reverse=True))
    underpopulation_list = dict(sorted(underpopulation_list.items(), key=lambda item: item[1][1], reverse=False))

    return overpopulation_list, underpopulation_list


def generate_jobs(date=datetime(year=2019, month=8, day=1, hour=1)):
    # Params for what counts as "bad" capacity
    optimal_capacity = 0.5
    upper_capacity = 0.7
    lower_capacity = 0.2

    # Get the maximum capacity of each station as dict
    station_caps = get_station_capacities()

    # Open the corrisponding predict file
    destination_directory = '../outputs/date_reports/'
    destination_file = destination_directory + date.strftime("%Y-%m-%d") + '.csv'
    if not os.path.exists(destination_file):
        print("File" + destination_file + " does not exist")
        return
    
    time = date.strftime("%#H:00")
    df = pandas.read_csv(destination_file,
                         usecols=['station', time])

    pop_dict = {}
    for index, row in df.iterrows():
        pop_dict[row['station']] = row[time]

    overpopulation_list, underpopulation_list = get_overunder_population(
        pop_dict,
        optimal_capacity,
        upper_capacity,
        lower_capacity)

    jobs = []

    while len(underpopulation_list) > 0 and len(overpopulation_list) > 0:
        num_to_move = min(overpopulation_list[list(overpopulation_list.keys())[0]][0],
                          underpopulation_list[list(underpopulation_list.keys())[0]][0])

        job = (
            list(overpopulation_list.keys())[0],
            list(underpopulation_list.keys())[0],
            num_to_move
        )

        jobs.append(job)

        pop_dict[job[0]] -= num_to_move
        pop_dict[job[1]] += num_to_move

        overpopulation_list, underpopulation_list = get_overunder_population(
            pop_dict,
            optimal_capacity,
            upper_capacity,
            lower_capacity)

    return jobs
