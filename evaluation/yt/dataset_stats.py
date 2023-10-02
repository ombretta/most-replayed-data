import json
import numpy as np
from moviepy.editor import VideoFileClip


def get_video_duration(video_path):
    try:
        clip = VideoFileClip(video_path)
        duration_seconds = clip.duration
        # clip.reader.close()
        # clip.audio.reader.close_proc()
        return duration_seconds
    except Exception as e:
        print(f"Error while processing {video_path}: {e}")
        return None


with open("/home/george/Documents/TUD/Thesis_video/PGL-SUM/data/datasets/splits/yt_500_splits.json") as f:
    splits = json.load(f)
    split = splits[0]
    video_ids = split["train_keys"] + split["test_keys"]
    
    durations = []
    for video_id in video_ids:
        video_path = f"/home/george/Documents/TUD/Thesis_video/YT/videos/{video_id}.mp4"
        duration = get_video_duration(video_path)
        durations.append(duration)
        # print(duration)

    durations = np.array(durations)
    print(durations.mean())
    print(durations.std())