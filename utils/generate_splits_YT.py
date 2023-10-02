import argparse
import csv
import numpy as np
import h5py
import re
import pandas as pd
import sklearn.model_selection
import json

parser = argparse.ArgumentParser()
parser.add_argument('csv', help="path to a .csv file with the video urls at column 0")
# parser.add_argument('--video_features_dir', required=True, help="path to the directory with video-id_$FEATURES_MODE.npy files")
# parser.add_argument('--features_mode', default='rgb')
# parser.add_argument('--video_heatmarkers_dir', required=True, help="path to the directory with heat-markers video-id.h5 files")
parser.add_argument("--n_splits", default=5, help="number of splits")
parser.add_argument('--out', help="path to .json output file", default='./data/splits/yt_splits.json')
args = parser.parse_args()

def indices_to_keys(keys, indices):
    return [keys[i] for i in indices]


with open(args.out, 'w') as out:
    with open(args.csv, 'r') as csv_file:
        video_ids = [re.search('(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})', row[0])[1] for row in csv.reader(csv_file)]
        kfold = sklearn.model_selection.KFold(n_splits=args.n_splits, shuffle=True)
        splits = kfold.split(video_ids)
        json.dump([{"train_keys": indices_to_keys(video_ids, train), "test_keys": indices_to_keys(video_ids, test)} for train, test in splits], out, indent=4)