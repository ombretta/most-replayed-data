import json
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from model.f1_score_test import interpolate_pred
from sklearn import metrics
import math


parser = argparse.ArgumentParser()
parser.add_argument('json_path', help="path to a .json file with results")
parser.add_argument('--video_heatmarkers_dir', required=True, help="path to the directory with heat-markers video-id.h5 files")
parser.add_argument('--histogram', action='store_true', help="Plot an histogram of the f1-scores instead of showing each video")
# parser.add_argument('--out', help="path to output h5 file", default='out.h5')
args = parser.parse_args()

video_dir = "../YT/videos"
f1_scores = []
n_features = []

plot_single = not args.histogram

with open(args.json_path, 'r') as f:
    n_shots = 100
    top_percent = 15
    pred = json.load(f)
    for video_id in pred.keys():
        try:
            heat_markers_pd = pd.read_hdf(args.video_heatmarkers_dir+'/'+video_id+'.h5')
        except:
            print(f"Skipping {video_id}, .h5 not found")
            continue
        current_pred = pred[video_id]
        pred_100 = interpolate_pred(np.array(current_pred), n_shots)
        partition_elem = int(math.floor(top_percent * n_shots / 100))
        pred_100_bool = np.zeros_like(pred_100)
        target_100_bool = np.zeros_like(heat_markers_pd.heatMarkerIntensityScoreNormalized)
        pred_100_bool[np.argpartition(pred_100, len(pred_100)-partition_elem-1)[-partition_elem:]] = 1
        target_100_bool[np.argpartition(heat_markers_pd.heatMarkerIntensityScoreNormalized, len(heat_markers_pd.heatMarkerIntensityScoreNormalized)-partition_elem-1)[-partition_elem:]] = 1

        f1_score = metrics.f1_score(target_100_bool, pred_100_bool)

        f1_scores.append(f1_score)
        n_features.append(len(current_pred))
        
        if f1_score < 0.1 : 
            print(video_id)

        if plot_single:
            fig = plt.figure()
            fig.suptitle(f"Video ID: {video_id}, f1 score {f1_score}")
            # gs = GridSpec(2, 10, figure=fig)
            gs = GridSpec(2,1, figure=fig)
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, :])
            ax1.set_title("Prediction")
            ax1.ticklabel_format(style='plain')
            ax1.xaxis.set_label('Time (s)')
            ax1.xaxis.set_major_formatter(lambda x, pos :  f"{int(x / 1000)}s")
            ax1.yaxis.set_label('Intensity score')
            ax1.set_ylim(0.0,1.0)
            ax1.plot(np.linspace(0, len(heat_markers_pd.timeRangeStartMillis)*heat_markers_pd.markerDurationMillis[0], len(current_pred)), current_pred, '-',  color='#00000080')
            ax1.bar(heat_markers_pd.timeRangeStartMillis+heat_markers_pd.markerDurationMillis/2, pred_100, width=heat_markers_pd.markerDurationMillis, color=['#00a6d6' if p == 0 else "green" if t==1 else "#e8710a" for (p,t) in zip(pred_100_bool, target_100_bool)])

            ax2.set_title("Ground truth")
            ax2.ticklabel_format(style='plain')
            ax2.xaxis.set_label('Time (ms)')
            ax2.yaxis.set_label('Intensity score')
            ax2.set_ylim(0.0,1.0)
            # ax2.bar(heat_markers_pd.timeRangeStartMillis, heat_markers_pd.heatMarkerIntensityScoreNormalized, width=heat_markers_pd.markerDurationMillis, color=['#00a6d6' if p == 0 else "#0084ab" for p in target_100_bool])
            ax2.bar(heat_markers_pd.timeRangeStartMillis, heat_markers_pd.heatMarkerIntensityScoreNormalized, width=heat_markers_pd.markerDurationMillis, color="#e80000")

            plt.show()

if args.histogram:
    fig = plt.figure()
    gs = GridSpec(3,1, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1:, :])
    ax1.hist(f1_scores, density=False, bins=np.linspace(0.0, 1.0, 16))
    ax1.set_title("F1-score distribution")
    ax1.ticklabel_format(style='plain')
    ax1.xaxis.set_label('F1-score')
    ax1.yaxis.set_label('Count')
    ax2.yaxis.set_label('Count')
    ax2.xaxis.set_label('F1-score')
    ax2.set_xlim(0.0,1.0)
    ax2.scatter(f1_scores, n_features)
    plt.show()