import sys
from typing import Iterator

import numpy as np
from tqdm import tqdm

from model import Video


def gaussian_model(video: Video, train_stop_frame: int, background_mean: np.ndarray, background_std: np.ndarray,
                   threshold: float = 2.5, total_frames: int = None) -> Iterator[np.ndarray]:
    for im, frame in tqdm(video.get_frames(train_stop_frame, -1), total=total_frames, file=sys.stdout):

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


def gaussian_model_adaptive(video: Video, train_stop_frame: int, background_mean: np.ndarray,
                            background_std: np.ndarray,
                            threshold: float = 2.5, rho: float = 0.1, total_frames: int = None) -> Iterator[np.ndarray]:
    for im, frame in tqdm(video.get_frames(train_stop_frame, -1), total=total_frames, file=sys.stdout):

        mask = np.zeros((im.shape[0], im.shape[1]))
        im_gray = np.mean(im, axis=-1) / 255
        row = 0
        for rows in im_gray:
            col = 0
            for pix in rows:
                if (abs(pix) - background_mean[row, col]) >= (threshold * (background_std[row, col] + 2)):
                    mask[row, col] = 1
                else:
                    background_mean = rho * pix + (1 - rho) * background_mean[row, col]
                    background_std = rho * pow((pix - background_mean[row, col]), 2) + (1 - rho) * background_std[
                        row, col]
                col += 1
            row += 1
        yield mask


def get_background_model(video: Video, train_stop_frame: int, total_frames: int = None) -> (np.ndarray, np.ndarray):
    background_list = None
    for im, frame in tqdm(video.get_frames(0, train_stop_frame), total=total_frames, file=sys.stdout):
        if background_list is None:
            background_list = np.zeros((im.shape[0], im.shape[1], train_stop_frame), dtype=np.int16)

        background_list[:, :, frame.id] = np.mean(im, axis=-1, dtype=np.int16)

    background_mean = np.mean(background_list, axis=-1) / 255
    background_std = np.std(background_list, axis=-1) / 255

    return background_mean, background_std
