from moviepy.editor import *
from pathlib import *
from random import randrange
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--duration", default=30, type=int, help="Final duration of the video")
parser.add_argument("--clip_duration", default=10, type=int, help="Final duration of the clips")

parser.add_argument("--num_clips", default=10, type=int, help="Number of clips")
parser.add_argument("--video_dir", default='../../../YT/videos')
parser.add_argument("--video_id", required=True)
parser.add_argument("-o", "--out_dir", required=True, help="Output directory")
parser.add_argument("--write_speedrun", action="store_true")
parser.add_argument("--write_clips", action="store_true")
parser.add_argument("--write_control_clip", action="store_true")
args = parser.parse_args()

out_dir = Path(args.out_dir)
os.makedirs(out_dir, exist_ok=True)

videos_path = Path(args.video_dir)
video_filename = args.video_id + ".mp4"
video_fullpath = videos_path.joinpath(video_filename)
video = VideoFileClip(video_fullpath.as_posix())

if args.write_speedrun:
    final_duration = args.duration
    final = video.fx(vfx.speedx, final_duration=final_duration)
    out_path = out_dir.joinpath(args.video_id + "_speedrun" + ".mp4")
    final.write_videofile(out_path.as_posix(), fps=5)

if args.write_clips:
    t_start = 0
    step = video.duration / args.num_clips
    for i in range(args.num_clips):
        clip = video.subclip(t_start, t_start+step)
        clip = clip.fx(vfx.speedx, final_duration=args.clip_duration)
        t_start += step
        out_path = out_dir.joinpath(args.video_id + "_clip_" + "{:02d}".format(i) + ".mp4")
        clip.write_videofile(out_path.as_posix(), fps=7.5)

if args.write_control_clip:
    step = video.duration / args.num_clips
    i = randrange(args.num_clips)
    t_start = i * step
    clip = video.subclip(t_start, t_start+step)
    clip = clip.fx(vfx.speedx, final_duration=args.clip_duration)
    thumb_duration = 0.5
    clip0 = clip.subclip(0, thumb_duration)
    clip1 = clip.subclip(thumb_duration)
    clip1_black = ColorClip(clip1.size, color=(0,0,0), duration=clip1.duration)
    
    # Generate a text clip 
    txt_clip = TextClip("Attention check:\nSelect CONTROL option", fontsize = 32, font = 'OpenSans', color = "white") 
        
    # setting position of text in the center and duration will be 10 seconds 
    txt_clip = txt_clip.set_pos('center').set_duration(clip1.duration)
    
    # Overlay the text clip on the first video clip 
    clip1_text = CompositeVideoClip([clip1_black, txt_clip])

    control_clip = concatenate([clip0, clip1_text])

    out_path = out_dir.joinpath(args.video_id + "_control" + ".mp4" ) # "_control_" + "{:02d}".format(i) + ".mp4")
    control_clip.write_videofile(out_path.as_posix(), fps=7.5)
