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

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def display():
    stations = get_station_names()

    if request.args.get('station_id') is None:
        return render_template('site.html', station_info=stations, result=None)
    else:

        if request.args.get('simple') is None:
            answer = full_predict(
                request.args.get('station_id'),
                request.args.get('int_time'),
                request.args.get('int_date'),
                request.args.get('int_day'),
                request.args.get('rain'),
                request.args.get('temp'),
                request.args.get('rhum')
            )
        elif request.args.get('simple') == 'on':
            answer = simple_predict(
                request.args.get('station_id'),
                request.args.get('int_time'),
                request.args.get('int_date'),
                request.args.get('int_day')
            )
        else:
            answer = 0

        return render_template('site.html', station_info=stations, result=answer)


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
    # print(inv_row[0][0])
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
    # print(inv_row[0][0])
    return inv_row[0][0]


if __name__ == '__main__':
    full_predict(2, 24, 213, 4, 0, 14, 87)
    # station_id, int_time, int_date, int_day, rain, temp, rhum
    app.run(debug=True)
