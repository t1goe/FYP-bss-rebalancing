import wget
import zipfile
import os
import csv

destination = './datasets/bss/dublin'

if not os.path.exists(destination):
    os.makedirs(destination)

base_url = "https://data.smartdublin.ie/dataset/33ec9fe2-4957-4e9a-ab55-c5e917c7a9ab/resource/8ddaeac6-4caf-4289-9835" \
           "-cf588d0b69e5/download/"
urls = ["dublinbikes_20200701_20201001",
        "dublinbikes_20200401_20200701",
        "dublinbikes_20200101_20200401",
        "dublinbikes_20191001_20200101",
        "dublinbikes_20190701_20191001",
        "dublinbikes_20190401_20190701",
        "dublinbikes_20190401_20190701",
        "dublinbikes_20181001_20190101",
        "dublinbikes_20180701_20181001",
        "dublin"]

for url in urls:
    if url + ".csv" in os.listdir(destination):
        print("File \"" + url + ".csv\" already exists")
        continue

    final_url = base_url + url + ".csv"
    print("starting download on " + url)
    wget.download(final_url, destination)
    print(url + " : downloaded")

print("Finished downloading Dublin BSS data")

# Remove generated tmp files
for file in os.listdir():
    if file[-4:] == ".tmp":
        os.remove(file)