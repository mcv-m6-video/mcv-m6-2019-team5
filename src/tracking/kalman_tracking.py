import cv2
import matplotlib.pyplot as plt
import numpy as np
from functional import seq
from matplotlib import patches

from model import Frame, SiameseDB
from .kalman import Sort, associate_detections_to_trackers


class KalmanTracking:

    def __init__(self):
        self.mot_tracker = Sort()  # create instance of the SORT tracker

    def __call__(self, frame: Frame, siamese: SiameseDB, debug=False):

        detections = seq(frame.detections).map(lambda d: d.to_sort_format()).to_list()
        detections = np.array(detections)
        trackers = self.mot_tracker.update(detections)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trackers)

        for match in matched:
            frame.detections[match[0]].id = int(trackers[match[1], 4])
            # print(match[0], " . ", match[1], " . ", frame.detections[match[0]].top_left, " . ", trackers[match[1]])

        for unmatched in unmatched_dets:
            if siamese is not None:
                new_id = siamese.query(frame.image, frame.detections[unmatched])
                if new_id != -1:
                    frame.detections[unmatched].id = new_id

        if debug:
            self.plot_tracking(frame)

    @staticmethod
    def plot_tracking(frame: Frame):
        plt.imshow(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        for det in frame.detections:
            rect = patches.Rectangle((det.top_left[0], det.top_left[1]), det.width, det.height,
                                     linewidth=1, edgecolor='blue', facecolor='none')
            plt.gca().add_patch(rect)

            plt.text(det.top_left[0] - 0, det.top_left[1] - 50, s='{}'.format(det.id),
                     color='white', verticalalignment='top',
                     bbox={'color': 'blue', 'pad': 0})
            plt.gca().add_patch(rect)
        plt.imshow(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        plt.close()
