from typing import List
import matplotlib.pyplot as plt
from functional import seq

from model import Frame


def iou_over_time(frames: List[Frame]):
    iou_per_frame = (seq(frames)
                     .map(lambda f: f.get_detection_iou_mean())
                     .to_list())

    plt.plot(iou_per_frame)
    plt.title('IoU over time')
    axes = plt.gca()
    axes.set_xlim((0, len(frames)))
    axes.set_ylim((0, 1))
    plt.show()
