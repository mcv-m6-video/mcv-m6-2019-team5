from typing import Iterator

import numpy as np

from model import Video


def gaussian_model(video: Video, train_stop_frame: int) -> Iterator[np.ndarray]:
    for im, frame in video.get_frames():
        if frame.id < train_stop_frame:
            # TODO train
            pass
        else:
            # TODO compute mask and return
            mask = frame
            yield mask
