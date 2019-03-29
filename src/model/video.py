import os
from typing import Iterator

import cv2
import numpy as np
from functional import seq


class Video:

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.files = (seq(os.listdir(video_path))
                      .map(lambda p: os.path.join(video_path, p))
                      .sorted()
                      .to_list())

    def get_frames(self, start: int = 0, end: int = None) -> Iterator[np.ndarray]:
        for i in range(start, len(self.files) if end is None else end):
            yield cv2.imread(os.path.join(self.video_path, 'frame_{:04d}.jpg'.format(i + 1)))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Video(path=\'video_path\')'
