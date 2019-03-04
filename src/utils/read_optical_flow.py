import cv2
import numpy as np


def read_optical_flow(path: str):
    I = cv2.imread('../datasets/optical_flow/detection/LKflow_000045_10.png')

    F_u = (I[:, :, 0] - 2 ^ 15) / 64
    F_v = (I[:, :, 1] - 2 ^ 15) / 64
    F_valid = I[:, :, 2]
    F_valid[F_valid > 1] = 1
    F_u[F_valid == 0] = 0
    F_v[F_valid == 0] = 0

    return np.dstack([
        F_u, F_v, F_valid
    ])
