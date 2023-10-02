import random
import json
import csv
import argparse

SPLIT_INDEX = 0
mode = 'test'
name = 'yt_500'

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out', required=True)
parser.add_argument('-n', '--num_videos', type=int, default=30)
args = parser.parse_args()


splits_filename = ['../../data/datasets/splits/' + name + '_splits.json']

with open(splits_filename[0]) as f:
    data = json.loads(f.read())
    for i, cur_split in enumerate(data):
        if i == SPLIT_INDEX:
            split = cur_split
            break

    video_names = split[mode + '_keys']
    random_videos = random.sample(video_names, args.num_videos)

    with open(args.out, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in random_videos:
            writer.writerow((row,))