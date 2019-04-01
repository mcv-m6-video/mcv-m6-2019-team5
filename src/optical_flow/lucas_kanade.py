import numpy as np
import cv2


class LucasKanade:

    def __init__(self, win_size=15, max_level=3):
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=500,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(win_size, win_size),
                              maxLevel=max_level,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __call__(self, im1: np.ndarray, im2: np.ndarray, p0: np.ndarray = None) -> np.ndarray:
        of = np.zeros((im1.shape[0], im1.shape[1], 2))
        old_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        if p0 is None:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)
        frame_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            b, a = new.ravel()
            d, c = old.ravel()
            of[int(a), int(b), :] = (b - d, a - c)
        return of
