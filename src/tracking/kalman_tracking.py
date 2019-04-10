import cv2
import matplotlib.pyplot as plt
import numpy as np
from functional import seq
from matplotlib import patches

from model import Frame, SiameseDB
from .kalman import Sort, associate_detections_to_trackers
from utils import IDGenerator
from matplotlib import colors
class KalmanTracking:

    def __init__(self):
        self.mot_tracker = Sort()  # create instance of the SORT tracker
        viridis = colors.ListedColormap(np.random.rand(256, 3))
        self.new_color = viridis(np.linspace(0, 1, 256))

    def __call__(self, frame: Frame, siamese: SiameseDB, debug=False, plot_number=False):

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
            else:
                frame.detections[unmatched].id = IDGenerator.next()

        if debug:
            self.plot_tracking_color(frame, plot_number)

    def plot_tracking_color(self, frame: Frame, plot_number):
        plt.imshow(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        for det in frame.detections:
            if det.id != -1:
                rect = patches.Rectangle((det.top_left[0], det.top_left[1]), det.width, det.height,
                                         linewidth=2, edgecolor=self.new_color[det.id, :], facecolor='none')
                plt.gca().add_patch(rect)
                if plot_number:
                    plt.text(det.top_left[0] - 0, det.top_left[1] - 50, s='{}'.format(det.id),
                             color='white', verticalalignment='top',
                             bbox={'color': 'blue', 'pad': 0})
                plt.gca().add_patch(rect)
        plt.imshow(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        plt.close()

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