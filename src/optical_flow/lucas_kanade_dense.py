import contextlib
import os
import cv2
import numpy as np
from scipy import signal


def lucas_kanade_dense(im1: np.ndarray, im2: np.ndarray, window_size=15, tau=1e-2) -> np.ndarray:
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
    w = window_size // 2  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(im1, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(im1, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(im2, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(im1, -kernel_t,
                                                                                          boundary='symm', mode=mode)

    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)
    # within window window_size * window_size
    for i in range(w, im1.shape[0] - w):
        for j in range(w, im1.shape[1] - w):
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
            b = np.reshape(It, (It.shape[0], 1))  # get b here
            A = np.vstack((Ix, Iy)).T  # get A here
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here
                u[i, j] = nu[0]
                v[i, j] = nu[1]

    return np.concatenate((u[..., None], v[..., None]), axis=2)
