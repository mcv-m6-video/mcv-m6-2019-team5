import numpy as np
import cv2


def opening(im: np.ndarray, kernel_side: int) -> np.ndarray:
    return cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((kernel_side, kernel_side)))


def closing(im: np.ndarray, kernel_side: int) -> np.ndarray:
    return cv2.morphologyEx(im, cv2.MORPH_CLOSE, np.ones((kernel_side, kernel_side)))
