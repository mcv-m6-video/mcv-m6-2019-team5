from typing import List
from matplotlib import patches
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import Detection, Rectangle
from utils import IDGenerator, show_optical_flow_arrows

INTERSECTION_THRESHOLD = 0.75


def overlap_flow_tracking(optical_flow_method,
                          im1: np.ndarray, det1: List[Detection],
                          im2: np.ndarray, det2: List[Detection], debug: bool = False):
    feature_params = dict(maxCorners=500,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    det1_flow = []
    if im1 is not None:
        mask = np.zeros((im1.shape[0], im1.shape[1]), dtype=np.uint8)
        for det in det1:
            mask[det.top_left[0]:det.top_left[0] + det.width, det.top_left[1]:det.top_left[1] + det.height] = 255
        if not debug:
            plt.imshow(mask, 'gray')
            plt.axis('off')
            plt.show()
            plt.close()

        p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), mask=mask, **feature_params)
        flow = optical_flow_method(im1, im2, p0)
        if not debug:
            show_optical_flow_arrows(im1, flow)

        for det in det1:
            det_flow = flow[det.top_left[0]:det.top_left[0] + det.width, det.top_left[1]:det.top_left[1] + det.height,
                       :]
            accum_flow = np.mean(det_flow[np.logical_or(det_flow[:, :, 0] != 0, det_flow[:, :, 1] != 0), :], axis=0)
            if np.isnan(accum_flow).any():
                accum_flow = (0, 0)
            det1_flow.append(
                Detection(det.id, det.label, (det.top_left[0] + accum_flow[0], det.top_left[1] + accum_flow[1]),
                          det.width, det.height))

        if debug:
            plt.figure()

        for det in det2:
            # if det1 is not None:
            _find_id(det, det1_flow, im2)

            if debug:
                rect = patches.Rectangle((det.top_left[1], det.top_left[0]), det.height, det.width,
                                         linewidth=2, edgecolor='blue', facecolor='none')
                plt.gca().add_patch(rect)
                plt.text(det.top_left[1], det.top_left[0], s='{} ~ {}'.format(det.label, det.id),
                         color='white', verticalalignment='top',
                         bbox={'color': 'blue', 'pad': 0})
        if debug:
            plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            # plt.savefig('../video/video_ssd_KalmanID/frame_{:04d}'.format(i))
            plt.show()
            plt.close()


def _find_id(det_new: Detection, dets_old: List[Detection], im2, debug: bool = False) -> None:
    for det in dets_old:
        if det_new.iou(det) > INTERSECTION_THRESHOLD:
            if debug:
                plt.figure()
                rect = patches.Rectangle((det.top_left[1], det.top_left[0]), det.height, det.width,
                                         linewidth=1, edgecolor='blue', facecolor='none')
                plt.gca().add_patch(rect)
                rect = patches.Rectangle((det_new.top_left[1], det_new.top_left[0]), det_new.height, det_new.width,
                                         linewidth=1, edgecolor='red', facecolor='none')
                plt.gca().add_patch(rect)
                plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
                plt.close()
            if det.id == -1:
                det.id = IDGenerator.next()
            det_new.id = det.id
            print(det.id)
            return
