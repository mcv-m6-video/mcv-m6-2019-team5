import fnmatch
import os
from typing import List

from model import Video


class Data:
    videos = List[Video]

    def __init__(self, video_names: List[str]):
        self.videos = []
        for video_name in video_names:
            self.videos.append(Video(video_name))
