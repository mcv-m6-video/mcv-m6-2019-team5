import cv2
import matplotlib.pyplot as plt
from matplotlib import patches

from model import Frame, Detection, SiameseDB
from utils import IDGenerator

INTERSECTION_THRESHOLD = 0.5


class OverlapTracking:
    """
    look_back: int. How many frames back to search for an intersection
    """

    def __init__(self, look_back=3):
        self.look_back = look_back
        self.prev_det = None

    def __call__(self, frame: Frame, siamese: SiameseDB, debug=False) -> None:
        for detection in frame.detections:
            self._find_id(detection)
            if detection.id == -1:
                if siamese is not None:
                    new_id = siamese.query(frame.image, detection)
                    if new_id != -1:
                        detection.id = new_id
                    else:
                        detection.id = IDGenerator.next()
        self.prev_det = frame.detections

        if debug:
            plt.imshow(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
            plt.axis('off')

            plt.subplot(1, 2, 1)
            for det in frame.detections:
                rect = patches.Rectangle((det.top_left[1], det.top_left[0]), det.height, det.width,
                                         linewidth=1, edgecolor='blue', facecolor='none')
                plt.gca().add_patch(rect)

                plt.text(det.top_left[0], det.top_left[1], s='{} ~ {}'.format(det.label, det.id),
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
