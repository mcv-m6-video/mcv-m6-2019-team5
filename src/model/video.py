from typing import Iterator, Tuple

import cv2
import numpy as np

from model import Frame


class Video:
    video_path: str
    car_only: bool

    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_frames(self, start_frame_number: int = 0) -> Iterator[Tuple[np.ndarray, Frame]]:
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

        num = 0
        while cap.isOpened():
            valid, image = cap.read()
            if not valid:
                break

            yield image, Frame(num)

            num += 1

        cap.release()
