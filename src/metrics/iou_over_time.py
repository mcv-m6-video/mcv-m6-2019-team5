from typing import List
import matplotlib.pyplot as plt
from functional import seq

from model import Frame


def iou_over_time(frames: List[Frame], ignore_classes=False):
    iou_per_frame = (seq(frames)
                     .map(lambda f: f.get_detection_iou_mean(ignore_classes))
                     .to_list())

    iou_gt = (
        seq(frames)
        .map(lambda f: 1 if len(f.ground_truth) > 0 else 0)
        .to_list()
    )

    plt.plot(iou_per_frame)
    plt.plot(iou_gt, 'r-')
    plt.title('IoU over time')
    axes = plt.gca()
    axes.set_xlim((0, len(frames)))
    axes.set_ylim((0, 1))
    plt.show()
