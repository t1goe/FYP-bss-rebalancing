from datetime import datetime

import math
import pickle
from os import listdir
from os.path import isfile, join
import csv

from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate

from generate_jobs import generate_jobs

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def display():
    stations = get_station_names()
    sorted(stations.keys(), key=lambda x: x.lower())
    if request.args.get('datetime') is not None:
        dt = request.args.get('datetime')

    if request.args.get('station_id') is None:
        return render_template('site.html', station_info=stations, result=None)
    else:

        if request.args.get('simple') is None:
            answer = full_predict(
                request.args.get('station_id'),
                convert_time(dt),
                convert_date(dt),
                convert_day(dt),
                request.args.get('rain'),
                request.args.get('temp'),
                request.args.get('rhum')
            )
        elif request.args.get('simple') == 'on':
            answer = simple_predict(
                request.args.get('station_id'),
                convert_time(dt),
                convert_date(dt),
                convert_day(dt)
            )
        else:
            answer = 0

        answer = round(answer)
        return render_template('site.html',
                               station_info=stations,
                               current_station_name=stations[request.args.get('station_id')],
                               current_time_info=request.args.get('datetime'),
                               result=answer)


@app.route('/list', methods=['POST', 'GET'])
def list():
    stations = get_station_names()

    sorted(stations.keys(), key=lambda x: x.lower())
    if request.args.get('date') is not None:
        dt = request.args.get('date')

    if request.args.get('date') is None:
        return render_template('list.html',
                               station_info=stations,
                               result=None)
    else:
        answers = {}
        for x in range(24):
            answers[str(x) + ":00"] = (
                round(
                    simple_predict(
                        request.args.get('station_id'),
                        (x * 12),
                        convert_date(dt + 'T00:00'),
                        convert_day(dt + 'T00:00')
                    )
                )
            )

    return render_template('list.html',
                           station_info=stations,
                           current_station_name=stations[request.args.get('station_id')],
                           current_date_info=request.args.get('date'),
                           result=answers
                           )


@app.route('/jobs', methods=['POST', 'GET'])
def jobs():
    if request.args.get('datetime') is None:
        return render_template('jobs.html',
                               current_date_info=None,
                               result=None
                               )
    else:
        input_date = datetime.strptime(request.args.get('datetime'), "%Y-%m-%dT%H:%M")
        results = generate_jobs(date=input_date)
        return render_template('jobs.html',
                               current_date_info=request.args.get('datetime'),
                               result=convert_results_to_english(results)
                               )


def convert_results_to_english(results):
    station_names = get_station_names()

    pos = 0
    for job in results:
        new_job = (
            str(job[0]) + " | " + station_names[str(job[0])],
            str(job[1]) + " | " + station_names[str(job[1])],
            job[2]
        )
        results[pos] = new_job
        pos += 1

    return results


def convert_time(x):
    """
    Converts TIME field in the CSV to an integer representing
    what time of day it is (in number of 5min increments) from 0 to 287
    eg
    - 00:00 -> 0
    - 00:10 -> 2
    - 02:20 -> 28
    etc
    """
    a = x.split('T')
    a = a[1].split(':')

    ans = math.floor((int(a[0]) * 12) + (int(a[1]) / 5))

    return ans


def convert_date(x):
    """
    Converts TIME field to an integer representing the day of the year

    eg
    - 2019-02-10 -> 41
    """
    current_date = datetime.strptime(x, "%Y-%m-%dT%H:%M")
    return current_date.strftime('%j')


def convert_day(x):
    """
    Converts TIME field to an integer representing the day of the week

    eg
    - 2019-02-10 -> 0 (Sunday)
    """
    current_date = datetime.strptime(x, "%Y-%m-%dT%H:%M")
    return current_date.strftime('%w')


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


def full_predict(station_id, int_time, int_date, int_day, rain, temp, rhum):
    destination_directory = '../datasets/bss/dublin/ml_models/'
    scaler_destination_directory = copy.deepcopy(destination_directory) + 'scalers/'

    model = tf.keras.models.load_model(destination_directory + 'station_' + str(station_id) + '.h5')

    file = open(scaler_destination_directory + 'station_' + str(station_id) + '.pkl', "rb")
    scaler = pickle.load(file)
    file.close()

    params = np.array([0, int_time, int_date, int_day, rain, temp, rhum])
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


if __name__ == '__main__':
    # full_predict(2, 24, 213, 4, 0, 14, 87)
    # station_id, int_time, int_date, int_day, rain, temp, rhum
    app.run(debug=True)
