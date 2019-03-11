from typing import Iterator, Tuple

import cv2
import numpy as np

from model import Frame


class Video:
    video_path: str
    car_only: bool

    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_frames(self, start: int = 0, end: int = -1) -> Iterator[Tuple[np.ndarray, Frame]]:
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        num = start
        while cap.isOpened():
            valid, image = cap.read()
            if not valid or (0 < end <= num):
                break

            yield image, Frame(num)

            num += 1

        cap.release()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Video(path=\'video_path\')'
