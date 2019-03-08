import sys
from typing import Iterator

import numpy as np
from tqdm import tqdm

from model import Video


def gaussian_model(video: Video, train_stop_frame: int, total_frames: int = None) -> Iterator[np.ndarray]:
    background_list = None
    background_mean = None
    background_std = None

    for im, frame in tqdm(video.get_frames(), total=total_frames, file=sys.stdout):
        if background_list is None:
            # backgroundMean = np.zeros((im.shape[0], im.shape[1]))
            background_list = np.zeros((im.shape[0], im.shape[1], train_stop_frame), dtype=np.int16)

        if frame.id < train_stop_frame:
            background_list[:, :, frame.id] = np.mean(im, axis=-1, dtype=np.int16)
            # backgroundMean = np.add(backgroundMean + im.mean(axis=-1, keepdims=0))
            if train_stop_frame - 1 == frame.id:
                background_mean = np.mean(background_list, axis=-1) / 255
                background_std = np.std(background_list, axis=-1) / 255

        else:
            threshold = 0.5
            mask = np.zeros((im.shape[0], im.shape[1]))
            im_gray = np.mean(im, axis=-1) / 255
            row = 0
            for rows in im_gray:
                col = 0
                for pix in rows:
                    if (abs(pix) - background_mean[row, col]) >= (threshold * (background_std[row, col] + 2)):
                        mask[row, col] = 1
                    col += 1
                row += 1
            yield mask
