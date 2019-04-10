import cv2
import matplotlib.pyplot as plt
from matplotlib import patches, colors

from model import Frame, Detection, SiameseDB
from utils import IDGenerator
import numpy as np

INTERSECTION_THRESHOLD = 0.55


class OverlapTracking:
    """
    look_back: int. How many frames back to search for an intersection
    """

    def __init__(self, look_back=3):
        self.look_back = look_back
        self.prev_det = None
        viridis = colors.ListedColormap(np.random.rand(256, 3))
        self.new_color = viridis(np.linspace(0, 1, 256))

    def __call__(self, frame: Frame, siamese: SiameseDB, debug=False, plot_number=False) -> None:
        for detection in frame.detections:
            self._find_id(detection)
            if detection.id == -1:
                if siamese is not None:
                    new_id = siamese.query(frame.image, detection)
                    if new_id != -1:
                        detection.id = new_id
                    else:
                        detection.id = IDGenerator.next()
                else:
                    detection.id = IDGenerator.next()
        self.prev_det = frame.detections

        if debug:
            self.plot_tracking_color(frame, plot_number)

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

    def _find_id(self, detection: Detection) -> None:
        if self.prev_det is None:
            return
        for detection2 in self.prev_det:
            if detection.iou(detection2) > INTERSECTION_THRESHOLD:
                detection.id = detection2.id
                break
