import os
from typing import Iterator, Tuple

import cv2
import numpy as np

from model import Frame


class Video:

    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_frames(self, start: int = 0, end: int = 2142) -> Iterator[Tuple[np.ndarray, Frame]]:
        for i in range(start+1, end+1):
            yield cv2.imread(os.path.join(self.video_path, 'frame_{:04d}.jpg'.format(i)))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Video(path=\'video_path\')'
