from typing import List
# from model import Frame
import ffmpeg
import numpy as np


class Video:
    # frames: List[Frame]

    def __init__(self, video_path: str, annotation_path: str):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        out, _ = (
            ffmpeg
                .input(video_path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True)
        )
        video = (
            np
                .frombuffer(out, np.uint8)
                .reshape([-1, height, width, 3])
        )
        print(video.shape)



video = Video("../../datasets/AICity_data/train/S03/c010/vdo.avi", "")
