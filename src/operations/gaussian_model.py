from typing import Iterator

import numpy as np
import matplotlib.pyplot as mp
from model import Video


def gaussian_model(video: Video, train_stop_frame: int) -> Iterator[np.ndarray]:

    for im, frame in video.get_frames():
        if frame.id == 0:
            backgroundMean = np.zeros((im.shape[0], im.shape[1]))
        if frame.id < train_stop_frame:
            imGray = im.mean(axis=-1, keepdims=0)
            col = 0
            for pixRow in imGray:
                row = 0
                for pix in pixRow:
                    meanValue = (backgroundMean[col, row] + pix) / ((row+1)*(col+1))
                    backgroundMean[col, row] = meanValue
                    row =+1
                col =+1
            pass
            backgroundMean
        else:
            # TODO compute mask and return
            mask = frame
            yield mask
