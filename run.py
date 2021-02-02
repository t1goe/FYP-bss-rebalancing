import pandas
import matplotlib.pyplot as plt
import data_dl


def convert_time(x):
    a = x.split(' ')
    a = a[1].split(':')

    ans = (int(a[0]) * 12) + (int(a[1]) / 5)

    return ans


def convert_date(x, start_date):
    a = x.split(' ')
    a = a[0].split('-')

    start = start_date.split(' ')
    start = start[0].split('-')


data_dl.dublin_weather()
data_dl.dublin_bss()

dataset = pandas.read_csv('./datasets/bss/dublin/dublinbikes_20200701_20201001.csv',
                          usecols=['STATION ID', 'TIME', 'AVAILABLE BIKES'])

date_list = []
for d in dataset['TIME'].unique():
    if not d.split(' ')[0] in date_list:
        date_list.append(d.split(' ')[0])

date_list.sort()

dataset['INT_TIME'] = dataset['TIME'].apply(lambda x: convert_time(x))
dataset['DATE'] = dataset['TIME'].apply(lambda x: x.split()[0])
print(dataset)

# print(dataset['STATION ID'].unique())
dataset = dataset.drop(dataset[dataset['STATION ID'] != 2].index)
for date in dataset['DATE'].unique():
    print('Processing date ' + str(date))
    temp_dataset = dataset.drop(dataset[dataset['DATE'] != date].index)
    # temp_dataset = temp_dataset.head(288)
    plt.plot(temp_dataset['INT_TIME'], temp_dataset['AVAILABLE BIKES'], label=("Date " + str(date)))

# for station in dataset['STATION ID'].unique()[:5]:
#     print('Processing station ' + str(station))
#     temp_dataset = dataset.drop(dataset[dataset['STATION ID'] != station].index)
#     temp_dataset = temp_dataset.head(288)
#     plt.plot(temp_dataset['INT_TIME'], temp_dataset['AVAILABLE BIKES'], label=("Station " + str(station)))

# dataset = dataset.drop(dataset[dataset['STATION ID'] != 2].index)
#
#
# dataset.drop(['STATION ID'], axis=1)
#
# print(dataset)
#
# plt.plot(dataset['INT_TIME'], dataset['AVAILABLE BIKES'])

plt.xlabel('Time')
plt.ylabel('Available bikes')

plt.show()
