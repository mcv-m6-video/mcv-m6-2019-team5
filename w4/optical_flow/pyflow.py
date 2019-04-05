import contextlib
import os

import numpy as np

from utils.suppress_stdout_stderr import suppress_stdout_stderr


def pyflow_optical_flow(im1: np.ndarray, im2: np.ndarray):
    import pyflow

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    with suppress_stdout_stderr():
        u, v, _ = pyflow.coarse2fine_flow(im1.astype(float) / 255, im2.astype(float) / 255, alpha, ratio)

    return np.concatenate((u[..., None], v[..., None]), axis=2)
