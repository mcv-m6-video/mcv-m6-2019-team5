from typing import Iterator

from tqdm import tqdm
from model import Video, Frame
import sys
import cv2 as cv
import numpy as np


def week2_soa_mod(video: Video, debug=False) -> Iterator[Frame]:
    th = 150
    fgbg = cv.createBackgroundSubtractorMOG2()
    for im, frame in tqdm(video.get_frames(int(2141 * 0.25)), total=int(2141 * 0.25) * 0.75, file=sys.stdout,
                          desc='Training model...'):

        fgmask = fgbg.apply(im)
        fgmask[fgmask < th] = 0
        kernel_e = np.ones((5, 5), np.uint8)
        kernel_d = np.ones((9, 9), np.uint8)
        diag = np.identity(5)
        t_diag = np.flip(diag, 0)
        kernel_d2 = np.uint8(np.logical_or(diag, t_diag))
        fgmask = cv.erode(fgmask, kernel_e)
        fgmask = cv.dilate(fgmask, kernel_d)
        fgmask = cv.dilate(fgmask, kernel_d2)
        cv.imshow('frame', fgmask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    cv.destroyAllWindows()
