import csv
import os
import sys
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
from datetime import datetime
import warnings


def get_index_of_date(df, date):
    # print(date)
    x = df.index[df['DATE'] == str(date).split(' ')[0]].tolist()
    if len(x) == 0:
        print("Date: " + str(date) + " not found in dataset")
        exit(1)

    return x[0]


def output_stats_to_csv(file_location, cols_used, mae, mse, rmse, r2):
    col_names = ['int_time', 'int_date', 'int_day', 'rain', 'temp', 'rhum', 'mae', 'mse', 'rmse', 'r2', 'timestamp']
    destination_directory = './datasets/bss/dublin/feature_optimization_stats/'
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    destination_file = destination_directory + file_location.split('/')[-1]

    if not os.path.exists(destination_file):

        with open(destination_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([col_names])

    with open(destination_file, 'a', newline='') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=col_names)

        writer.writerow({
            'int_time': ('int_time' in cols_used),
            'int_date': ('int_date' in cols_used),
            'int_day': ('int_day' in cols_used),
            'rain': ('rain' in cols_used),
            'temp': ('temp' in cols_used),
            'rhum': ('rhum' in cols_used),
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        })


def train_model(file_location,
                train_start_date=datetime(year=2018, month=8, day=1),
                train_end_date=datetime(year=2019, month=7, day=30),
                test_start_date=datetime(year=2019, month=8, day=1),
                test_end_date=datetime(year=2019, month=12, day=31),
                cols_to_use=None,
                verbose=1
                ):
    if cols_to_use is None:
        cols_to_use = ['int_time', 'int_date', 'int_day']

    cols_to_use.insert(0, 'AVAILABLE BIKES')
    cols_to_use.insert(0, 'TIME')
    # load dataset
    dataset = read_csv(file_location, usecols=cols_to_use)
    dataset['DATE'] = dataset['TIME'].apply(lambda x: x.split(' ')[0])

    if 'rain' in cols_to_use:
        dataset = dataset[dataset['rain'].str.strip().astype(bool)]

    train_start_index = (get_index_of_date(dataset, train_start_date))
    train_end_index = (get_index_of_date(dataset, train_end_date))
    # print( train_end_index - train_start_index)

    test_start_index = (get_index_of_date(dataset, test_start_date))
    test_end_index = (get_index_of_date(dataset, test_end_date))
    # print(test_end_index - test_start_index)

    dataset = dataset.drop(['TIME', 'DATE'], axis=1)
    # print(dataset.head())
    # print(dataset)
    values = dataset.values
    # print(values.shape)

    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    # print(values.shape)
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = scaled

    # print(scaled)

    # split into train and test sets
    # values = reframed.values

    train = scaled[train_start_index:train_end_index, :]
    test = scaled[test_start_index:test_end_index, :]
    # train = values[train_start:train_end, :]
    # test = values[test_start:test_end, :]

    # split into input and outputs
    train_X, train_y = train[:, 1:], train[:, 0]
    test_X, test_y = test[:, 1:], test[:, 0]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')


    # fit network
    history = model.fit(train_X, train_y,
                        epochs=150,
                        batch_size=72,
                        validation_data=(test_X, test_y),
                        verbose=verbose,
                        shuffle=False)
    # plot history
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # pyplot.show()

    # make a prediction
    yhat = model.predict(test_X)
    # print(test_X)
    # print(yhat)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast

    inv_yhat = concatenate((yhat, test_X), axis=1)
    # print(yhat.shape)
    # print(test_X.shape)
    # print(inv_yhat.shape)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE

    # np.set_printoptions(threshold=sys.maxsize)
    # temp = concatenate((inv_y, inv_yhat))
    # print(temp)
    # print(inv_y)
    # print(inv_yhat)

    # print()
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    mae = mean_absolute_error(inv_y, inv_yhat)
    mse = mean_squared_error(inv_y, inv_yhat)
    r2 = r2_score(inv_y, inv_yhat)
    print('Test MAE: %.3f' % mae)
    print('Test MSE: %.3f' % mse)
    print('Test RMSE: %.3f' % rmse)
    print('Test R2: %.30f' % r2)

    output_stats_to_csv(file_location, cols_to_use, mae, mse, rmse, r2)


def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


def test_powersets(start_position=0,
                   file='./datasets/bss/dublin/reorg_plus_weather/station_2.csv',
                   train_start_date=None,
                   train_end_date=None,
                   test_start_date=None,
                   test_end_date=None,
                   ):
    attr_list = [
        'int_time',
        'int_date',
        'int_day',
        'rain',
        'temp',
        'rhum'
    ]

    y = list(powerset(attr_list))
    # y.sort()
    print(len(y))
    y = sorted(y, key=len)
    y.pop(0)

    for x in y[start_position:]:
        print(str(start_position) + "/" + str(len(y) - 1))
        start_position = start_position + 1
        print(x)
        warnings.filterwarnings("ignore")
        if train_start_date is not None and train_end_date is not None and test_start_date is not None and test_end_date is not None:
            train_model(file,
                        train_start_date=train_start_date,
                        train_end_date=train_end_date,
                        test_start_date=test_start_date,
                        test_end_date=test_end_date,
                        cols_to_use=x,
                        verbose=0)
        else:
            train_model(file,
                        cols_to_use=x,
                        verbose=0)
        print()
        keras.backend.clear_session()


test_powersets()
