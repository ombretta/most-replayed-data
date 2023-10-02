import numpy as np
import cv2
from moviepy.editor import *

def generate_video(video_path, video_output_path, summary, backend = 'moviepy'):
    """
    :param str backend: moviepy (default) includes audio, cv2 is more precise but has no audio
    """
    if backend == 'moviepy':
        video = VideoFileClip(str(video_path))
        fps = video.fps
        duration = video.duration
        subclips = []
        frame_idx = 0
        start_frame = 0
        while(True):
            while(frame_idx < len(summary) and summary[frame_idx] == 0):
                frame_idx +=1    
                start_frame = frame_idx
            while(frame_idx < len(summary) and summary[frame_idx] == 1):
                frame_idx +=1
            if(frame_idx > start_frame):
                print(f"appended new subclip [{start_frame+1},{frame_idx-1}]")
                subclips.append(video.subclip(start_frame / fps, (frame_idx-1) / fps))

            if(frame_idx >= len(summary) or frame_idx / fps >= duration):
                break

        out = concatenate_videoclips(subclips)

        #writing the video into a file / saving the combined video
        out.write_videofile(str(video_output_path))
        video.close()

    elif backend == 'cv2':
        video = cv2.VideoCapture(str(video_path), apiPreference=cv2.CAP_FFMPEG)
        fps = round(video.get(cv2.CAP_PROP_FPS))

        if (video.isOpened() == False):
            print("Error reading video file")
            return

        # We need to set resolutions.
        # so, convert them from float to integer.
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        size = (frame_width, frame_height)

        print(video_path)
        print(f"fps: {fps}, size: ({frame_width}, {frame_height})")

        #cv2.VideoWriter(filename, fourcc, fps, frameSize)

        video_output_path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(video_output_path),
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps, size)
        frame_idx = 0

        while(True):
            ret, frame = video.read()

            if ret != True:
                break

            if summary[frame_idx] == 1:
                    # Output the frame
                    out.write(frame)
                    # Display the frame
                    # cv2.imshow('Frame', frame)           

            frame_idx += 1

        # When everything done, release
        # the video capture and video
        # write objects    
        cv2.destroyAllWindows()
        video.release()
        out.release()

        print(f"Wrote video summmary to {video_output_path}")

