from typing import Iterator

import numpy as np
import matplotlib.pyplot as mp
from model import Video


def gaussian_model(video: Video, train_stop_frame: int) -> Iterator[np.ndarray]:
    for im, frame in video.get_frames():
        if frame.id == 0:
            #backgroundMean = np.zeros((im.shape[0], im.shape[1]))
            backgroundList = np.zeros((im.shape[0], im.shape[1], train_stop_frame))
        if frame.id < train_stop_frame:
            backgroundList[:, :, frame.id] = np.mean(im, axis=-1)/255
            #backgroundMean = np.add(backgroundMean + im.mean(axis=-1, keepdims=0))
            if train_stop_frame-1 == frame.id:
                backgroundMean = np.mean(backgroundList, axis=-1)
                backgroundStd = np.std(backgroundList, axis=1)

        else:
            thershold=0.5
            mask = np.zeros((im.shape[0], im.shape[1]))
            imGray = np.mean(im, axis=-1)/255
            row = 0
            for rows in imGray:
                col = 0
                for pix in rows:
                    if (abs(pix)-backgroundMean[row,col])>= (thershold * (backgroundStd[row,col]+2)):
                        mask[row, col] = 1
                    col +=1
                row +=1
            yield mask
