import numpy as np
import cv2
import numpy as np


def gunner_farneback(im1: np.ndarray, im2: np.ndarray, debug: bool = False) -> np.ndarray:
    prvs = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    if debug:
        hsv = np.zeros_like(im1)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2', rgb)
        cv2.waitKey(0)

    return flow

