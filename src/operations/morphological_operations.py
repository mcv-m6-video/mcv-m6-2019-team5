import numpy as np
import cv2


def opening(im: np.ndarray, kernel_side: int) -> np.ndarray:
    return cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((kernel_side, kernel_side), dtype=np.uint8))


def closing(im: np.ndarray, kernel_side: int) -> np.ndarray:
    return cv2.morphologyEx(im, cv2.MORPH_CLOSE, np.ones((kernel_side, kernel_side), dtype=np.uint8))


def dilate(im: np.ndarray, kernel_side: int) -> np.ndarray:
    return cv2.dilate(im, kernel_side)
