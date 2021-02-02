import os
import zipfile
import wget


###
# Download dublin weather data
###
def dublin_weather():
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


###
# Download dublin bss data
###
def dublin_bss():
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

    for url in urls:
        filename = url.split("/")[-1]
        if filename in os.listdir(destination):
            print("File \"" + filename + "\" already exists")
            continue

        final_url = base_url + url
        print("starting download on " + filename)
        wget.download(final_url, destination)
        print(filename + " : downloaded\n")

    print("Finished downloading Dublin BSS data")

    # Clear generated tmp files
    for file in os.listdir():
        if file[-4:] == ".tmp":
            os.remove(file)
