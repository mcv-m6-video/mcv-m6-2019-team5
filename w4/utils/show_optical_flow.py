import numpy as np
import matplotlib.pyplot as plt
import cv2


def show_optical_flow(im1: np.ndarray, flow: np.ndarray):
    hsv = np.zeros_like(im1)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    plt.figure()
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()
