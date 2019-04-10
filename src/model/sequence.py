import os
from typing import Iterable

from model import Video


class Sequence:

    def __init__(self, sequence_path: str):
        self.sequence_path = sequence_path

    def get_videos(self) -> Iterable[Video]:
        parent_dir = os.listdir(self.sequence_path)
        parent_dir.sort()
        for video_path in parent_dir:
            yield Video(os.path.join(self.sequence_path, video_path))

    def __repr__(self):
        return self.__str__()
