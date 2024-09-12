import requests
import os
import wget
import sys


def download(path, fileName):
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(f"data/{fileName}"):
        wget.download(path, f"data/{fileName}", bar=bar_progress)
    else:
        print("File already exists")


def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] Mbytes" % (current / total * 100, current/1000000, total/1000000)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


newest_data_url = "https://data.police.uk/data/archive/latest.zip"
final_url = requests.head(newest_data_url, allow_redirects=True).url
filename = final_url.split('/')[-1]

print(filename)
download(newest_data_url, filename)
