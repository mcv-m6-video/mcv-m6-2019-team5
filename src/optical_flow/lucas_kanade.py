import numpy as np
from scipy import signal
import cv2


def lucas_kanade(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    window_size = 9
    tau = 1e-2

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])
    w = window_size // 2  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window

    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(im1_gray, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(im1_gray, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(im2_gray, kernel_t, boundary='symm', mode=mode) + \
         signal.convolve2d(im1_gray, -kernel_t, boundary='symm', mode=mode)

    u = np.zeros(im1_gray.shape)
    v = np.zeros(im1_gray.shape)
    for i in range(w, im1_gray.shape[0] - w):
        for j in range(w, im1_gray.shape[1] - w):
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
            b = np.reshape(It, (It.shape[0], 1))
            A = np.vstack((Ix, Iy)).T

            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b)
                u[i, j] = nu[0]
                v[i, j] = nu[1]

    return np.concatenate((u[..., None], v[..., None]), axis=2)
