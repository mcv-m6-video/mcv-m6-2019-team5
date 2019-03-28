from typing import List
import matplotlib.pyplot as plt
from functional import seq

from model import Frame


def iou_over_time(frames: List[Frame], ignore_classes=False, show=True):
    iou_per_frame = (seq(frames)
                     .map(lambda f: f.get_detection_iou_mean(ignore_classes))
                     .to_list())

    iou_gt = (
        seq(frames)
        .map(lambda f: 1 if len(f.ground_truth) > 0 else 0)
        .to_list()
    )

    plt.plot(range(len(frames)), iou_per_frame, 'b-', label='IoU')
    plt.plot(range(len(iou_gt)), iou_gt, 'r-', label='Best possible IoU')
    if show:
        plt.title('IoU over time')
        axes = plt.gca()
        axes.set_xlim((0, len(frames)))
        axes.set_ylim((0, 1))
        plt.show()
