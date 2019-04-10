from typing import List

import cv2
import numpy as np

from model import Frame, Detection


class RemoveParkedCars:
    def __init__(self, win_size=15, max_level=3):
        self.prev_frame = None
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(win_size, win_size), maxLevel=max_level,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __call__(self, frame: Frame) -> List[Detection]:
        mot_detections = []
        if self.prev_frame is not None:
            flow = self._optical_flow(frame.image)
            for det in self.prev_frame.detections:
                if det.id == -1:
                    continue

                det2 = None
                for new_det in frame.detections:
                    if new_det.id == det.id:
                        det2 = new_det
                        break

                if det2 is None:
                    continue

                det_flow = flow[det.top_left[1]:det.top_left[1] + det.height,
                                det.top_left[0]:det.top_left[0] + det.width, :]
                mean_flow = (0, 0)
                non_zero_values = det_flow[np.logical_or(det_flow[:, :, 0] != 0, det_flow[:, :, 1] != 0), :]
                if non_zero_values.size > 0:
                    mean_flow = np.mean(non_zero_values, axis=0)

                if np.linalg.norm(mean_flow) > 1.25:
                    mot_detections.append(det2)

        self.prev_frame = frame
        return mot_detections

    def _optical_flow(self, image) -> np.ndarray:
        of = np.zeros((image.shape[0], image.shape[1], 2))
        p0 = self._get_features()
        p1, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(self.prev_frame.image, cv2.COLOR_BGR2GRAY),
                                               cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), p0, None, **self.lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            b, a = new.ravel()
            d, c = old.ravel()
            of[int(c), int(d), :] = (b - d, a - c)
        return of

    def _get_features(self):
        mask = np.zeros((self.prev_frame.image.shape[0], self.prev_frame.image.shape[1]), dtype=np.uint8)
        for det in self.prev_frame.detections:
            mask[det.top_left[1]:det.top_left[1] + det.height, det.top_left[0]:det.top_left[0] + det.width] = 255
        p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(self.prev_frame.image, cv2.COLOR_BGR2GRAY), mask=mask,
                                     **self.feature_params)
        return p0
