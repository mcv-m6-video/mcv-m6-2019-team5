from tqdm import tqdm
from model import Video
import sys
import cv2 as cv


def week2_soa(video: Video):
    fgbg = cv.createBackgroundSubtractorMOG2()
    for im, frame in tqdm(video.get_frames(int(2141 * 0.25)), total=int(2141 * 0.25) * 0.75, file=sys.stdout,
                          desc='Training model...'):

        fgmask = fgbg.apply(im)
        cv.imshow('frame', fgmask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    cv.destroyAllWindows()
