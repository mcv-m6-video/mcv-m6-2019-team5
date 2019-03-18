import sys
from typing import List

from functional import seq
from tqdm import tqdm

from model import Video, Frame, Rectangle, Detection
import cv2
import time
import numpy as np

from operations import Sort
from utils import read_annotations, read_detections


def object_tracking_kf(frames: List[Frame]) -> List[Sort]:
    #    total_time = 0.0
    #    total_frames = 0
    out = []

    mot_tracker = Sort()  # create instance of the SORT tracker
    for frame in frames:  # all frames in the sequence
        dets = frame.get_format_detections()
        dets = np.array(dets)
        #        total_frames += 1
        #        start_time = time.time()
        trackers = mot_tracker.update(dets)
        #        cycle_time = time.time() - start_time
        #       total_time += cycle_time

        out.append(trackers)
    return out


#    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))


start_frame = 1440
end_frame = 1789

video = Video("../../datasets/AICity_data/train/S03/c010/vdo.avi")

gt = read_annotations('../../annotations', start_frame, end_frame)

alg = 'mask_rcnn'

detections = read_detections('../../datasets/AICity_data/train/S03/c010/det/det_{0}.txt'.format(alg))
detections = detections[start_frame: end_frame + 1]

frames = []

for i in range(start_frame, end_frame):
    f = Frame(i)
    f.ground_truth = gt[i-start_frame]
    f.detections = detections[i-start_frame]
    frames.append(f)
    i += 1
object_tracking_kf(frames)
