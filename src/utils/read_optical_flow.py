import cv2
import numpy as np


def read_optical_flow(path: str):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    im = np.flip(im, axis=2).astype(np.uint16)

    f_u = (im[:, :, 0] - 2 ^ 15) / 64
    f_v = (im[:, :, 1] - 2 ^ 15) / 64
    f_valid = im[:, :, 2]
    f_valid[f_valid > 1] = 1
    f_u[f_valid == 0] = 0
    f_v[f_valid == 0] = 0

    return np.dstack([
        f_u, f_v, f_valid
    ])
