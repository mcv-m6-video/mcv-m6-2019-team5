import os
from collections import Iterator

from model import Video


class Sequence:

    def __init__(self, sequence_path: str):
        self.sequence_path = sequence_path

    def get_videos(self):
        parent_dir = os.listdir(self.sequence_path)
        for video_path in parent_dir:
            yield Video(video_path)

    def __repr__(self):
        return self.__str__()
