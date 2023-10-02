import requests as req
import json
import math
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from PGLSUM.inference.knapsack_implementation import knapSack
from PGLSUM.evaluation.generate_video import generate_video

from moviepy.editor import *
import youtube_dl

VIDEO_PATH = "./videos"
HEAT_MARKERS_PATH = "./heat_markers"
# https://github.com/ytdl-org/youtube-dl/blob/3e4cedf9e8cd3157df2457df7274d0c842421945/youtube_dl/YoutubeDL.py#L137-L312
def dwl_vid(url):
    ydl_opts = {"format": "mp4[height=240]", "outtmpl": f"{VIDEO_PATH}/%(id)s.%(ext)s"}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def cubic_spline(heat_markers):
    return scipy.interpolate.CubicSpline(heat_markers.timeRangeStartMillis+heat_markers.markerDurationMillis, heat_markers.heatMarkerIntensityScoreNormalized, axis=0, bc_type='natural', extrapolate=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("id", help="ID of the video to download")
    parser.add_argument("--generate-video", action='store_true')
    parser.add_argument("--no-save", action='store_true')
    args = parser.parse_args()
    video_id = args.id
else:   
    # video_id = "hYRkQEFWnNo"
    video_id = "K43yCQwwCxQ" # dopest vlog ever
    # video_id = "6YeyU62aGNY"
    # video_id = "qDfwWZaNtSI"

YT_OPERATIONAL_API_URL = "http://localhost/YouTube-operational-API/" # "https://yt.lemnoslife.com"

r = req.get(YT_OPERATIONAL_API_URL+'/videos?part=mostReplayed&id={}'.format(video_id))

if not hasattr(r, "text") or not isinstance(r.text, str) or len(r.text) == 0:
    print("This video does not provide Most Replayed data")
    exit(-1)
# print(r.text)
json_response = json.loads(r.text)

most_replayed = json_response["items"][0]["mostReplayed"]

if most_replayed is None:
    print("This video does not provide Most Replayed data")
    exit(-1)

fig = plt.figure()
# gs = GridSpec(2, 10, figure=fig)
gs = GridSpec(2,1, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])

fig.suptitle(f"Most Replayed data for {video_id}")


heat_markers = most_replayed["heatMarkers"]
print(heat_markers)
heat_markers = [m["heatMarkerRenderer"] for m in heat_markers]

heat_markers = pd.DataFrame(heat_markers)

if not args.no_save:
    heat_markers.to_hdf(f"heat_markers/{video_id}.h5", "/heat_markers", mode='w')

print(f"number of heat markers: {len(heat_markers)}")
last_marker = heat_markers.tail(1)
last_time = int(last_marker.timeRangeStartMillis)
duration_ms = int(last_time + last_marker.markerDurationMillis)
print(f"duration: {duration_ms}")
xs = np.arange(0, duration_ms, 1)
heat_markers_spline = cubic_spline(heat_markers)
#ax1
ax1.set_title("Cubic interpolated")
ax1.ticklabel_format(style='plain')
ax1.xaxis.set_label('Time (s)')
ax1.xaxis.set_major_formatter(lambda x, pos :  f"{int(x / 1000)}s")
ax1.yaxis.set_label('Intensity score')
ax1.plot(xs, heat_markers_spline(xs))
#ax2
ax2.set_title("Raw data")
ax2.ticklabel_format(style='plain')
ax2.xaxis.set_label('Time (ms)')
ax2.yaxis.set_label('Intensity score')
ax2.bar(heat_markers.timeRangeStartMillis, heat_markers.heatMarkerIntensityScoreNormalized, width=heat_markers.markerDurationMillis)

if args.generate_video:
    ### CONSTANTS
    # one scene every 2 seconds
    SHOT_DURATION_MS = 2000
    # ratio of summary duration to video duration (default 15%)
    SUMMARY_DURATION_RATIO = 0.15

    x_integral = np.linspace(0, duration_ms, 10000)
    shots_bins = np.arange(0, duration_ms, SHOT_DURATION_MS)
    x_pdf = heat_markers_spline(x_integral)
    bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(x_integral, x_pdf, statistic='mean', bins=shots_bins)

    shots_durations = bin_edges[1:]-bin_edges[:-1]
    shots_durations = shots_durations.astype(int)
    shots_selected = knapSack(math.floor(SUMMARY_DURATION_RATIO * duration_ms), shots_durations, bin_means, len(bin_means))
    print(shots_selected)

    # Providing x and y label to the chart
    ax1.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors=['r' if i in shots_selected else 'g' for i in range(len(bin_means))], lw=5, label='binned mean of data')




# plt.show()
# create video summary with moviepy

if not args.no_save:
    video_path = f"{VIDEO_PATH}/{video_id}.mp4"
    # youtube_dl download can take long for new videos
    dwl_vid(video_id)
    video = VideoFileClip(str(video_path))
    if args.generate_video:
        summary_path = f"{VIDEO_PATH}/{video_id}_{SUMMARY_DURATION_RATIO}.mp4"
        fps = video.fps
        duration = video.duration # float
        n_frames = int(duration * fps)
        summary = np.zeros([n_frames])
        conv_ratio = fps / 1000.0
        for i in shots_selected:
            start_frame = int(bin_edges[i] * conv_ratio)
            end_frame = int(bin_edges[i+1] * conv_ratio)
            summary[start_frame:end_frame] = 1
        generate_video(video_path, summary_path, summary)



# video frame extraction

# KPS = 1# Target Keyframes Per Second
# VIDEO_PATH = "video1.avi"#"path/to/video/folder" # Change this
# IMAGE_PATH = "images/"#"path/to/image/folder" # ...and this 
# EXTENSION = ".png"
# cap = cv2.VideoCapture(VIDEO_PATH)
# fps = round(cap.get(cv2.CAP_PROP_FPS))
# print(fps)
# # exit()
# hop = round(fps / KPS)
# curr_frame = 0
# while(True):
#     ret, frame = cap.read()
# ifnot ret: break
# if curr_frame % hop == 0:
#         name = IMAGE_PATH + "_" + str(curr_frame) + EXTENSION
#         cv2.imwrite(name, frame)
#     curr_frame += 1
# cap.release()

# plot percentile frames compilation
plt.show()

