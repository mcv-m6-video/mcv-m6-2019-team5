import sys
from enum import Enum
from typing import Iterator

import cv2
import numpy as np
from tqdm import tqdm

from model import Video
from utils.memory import memory


class PixelValue(Enum):
    GRAY = 0
    HUE = 1


def gaussian_model(video: Video, frame_start: int, background_mean: np.ndarray, background_std: np.ndarray,
                   alpha: float = 2.5, pixel_value: PixelValue = PixelValue.GRAY,
                   total_frames: int = None) -> Iterator[np.ndarray]:
    for im, frame in tqdm(video.get_frames(frame_start), total=total_frames, file=sys.stdout,
                          desc="Non-adaptive gaussian model..."):

        if pixel_value == PixelValue.GRAY:
            im_values = np.mean(im, axis=-1) / 255
        elif PixelValue.HUE:
            im_values = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 0] / 255
        else:
            raise Exception

        mask = (np.abs(im_values) - background_mean) >= (alpha * (background_std + (5 / 255)))

        yield mask.astype(np.uint8) * 255


def gaussian_model_adaptive(video: Video, train_stop_frame: int, background_mean: np.ndarray,
                            background_std: np.ndarray,
                            alpha: float = 2.5, rho: float = 0.1, pixel_value: PixelValue = PixelValue.GRAY,
                            total_frames: int = None) -> Iterator[np.ndarray]:
    for im, frame in tqdm(video.get_frames(train_stop_frame, -1), total=total_frames, file=sys.stdout,
                          desc='Adaptive gaussian model...'):

        if pixel_value == PixelValue.GRAY:
            im_values = np.mean(im, axis=-1) / 255
        elif PixelValue.HUE:
            im_values = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 0] / 255
        else:
            raise Exception

        mask = (np.abs(im_values) - background_mean) >= (alpha * (background_std + 5 / 255))
        background_mean = rho * im_values + (1 - rho) * background_mean
        background_std = np.sqrt(
            rho * np.power((im_values - background_mean), 2) + (1 - rho) * np.power(background_std, 2)
        )

        yield mask.astype(np.uint8) * 255


@memory.cache
def get_background_model(video: Video, train_stop_frame: int, total_frames: int = None) -> (np.ndarray, np.ndarray):
    background_list = None
    for im, frame in tqdm(video.get_frames(0, train_stop_frame), total=total_frames, file=sys.stdout,
                          desc='Training model...'):
        if background_list is None:
            background_list = np.zeros((im.shape[0], im.shape[1], train_stop_frame), dtype=np.int16)

        background_list[:, :, frame.id] = np.mean(im, axis=-1, dtype=np.int16)

    background_mean = np.mean(background_list, axis=-1) / 255
    background_std = np.std(background_list, axis=-1) / 255

    return background_mean, background_std
