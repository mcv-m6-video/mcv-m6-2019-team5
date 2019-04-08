from functional import seq
from model import Frame
from tracking import Sort, associate_detections_to_trackers
import numpy as np
import cv2


class KalmanTracking:

    def __init__(self):
        self.mot_tracker = Sort()  # create instance of the SORT tracker

    def __call__(self, frame: Frame, frames=None, debug=False, *args):

        detections = seq(frame.detections).map(lambda d: d.to_sort_format()).to_list()
        detections = np.array(detections)
        trackers = self.mot_tracker.update(detections)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trackers)

        for match in matched:
            frame.detections[match[0]].id = int(trackers[match[1], 4])
            # print(match[0], " . ", match[1], " . ", frame.detections[match[0]].top_left, " . ", trackers[match[1]])

        if debug:
            grid = np.array(self.video[frame.id])
            for d in frame.detections:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(grid, str(d.id), (int(d.top_left[1]), int(d.top_left[0])), font, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.rectangle(grid, (int(d.top_left[1]) + 5, int(d.top_left[0]) + 5),
                              (int(d.get_bottom_right()[1]), int(d.get_bottom_right()[0])), (0, 0, 255), thickness=5)
            cv2.imshow('f', grid)
            cv2.waitKey()


def track_detections():

    kalman = KalmanTracking()
    for video in Sequence("../datasets/AICity_data/train/S03"):
        for frame in video:

            kalman(frame)
            print(seq(frame.detections).map(lambda d: d.id).to_list())

