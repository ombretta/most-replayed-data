import argparse
import csv
import numpy as np
import h5py
import re
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('csv', help="path to a .csv file with the video urls at column 0")
parser.add_argument('--video_features_dir', required=True, help="path to the directory with video-id_$FEATURES_MODE.npy files")
parser.add_argument('--features_mode', default='rgb')
parser.add_argument('--video_heatmarkers_dir', required=True, help="path to the directory with heat-markers video-id.h5 files")
parser.add_argument('--out', help="path to output h5 file", default='out.h5')
args = parser.parse_args()

with h5py.File(args.out, 'w') as hf:
    with open(args.csv, 'r') as csv_file:
        for row in csv.reader(csv_file):
            video_url = row[0]
            match = re.search('(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})', video_url)
            video_id = match[1]
            print(match[1])
            video_group = hf.create_group(video_id)
            try:
                features = np.load(args.video_features_dir+'/'+video_id+'_'+args.features_mode+'.npy')
                heat_markers_pd = pd.read_hdf(args.video_heatmarkers_dir+'/'+video_id+'.h5')
                heat_markers = np.array(heat_markers_pd["heatMarkerIntensityScoreNormalized"])
                print("features:", features.shape)
                print("heat_markers:", heat_markers.shape)
                video_group.create_dataset('features', data=features)
                video_group.create_dataset('heat-markers', data=heat_markers)
            except Exception as e:
                print(f"Skipping video {video_id}")
                print(e)
                continue
