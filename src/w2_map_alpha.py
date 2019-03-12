import sys

from methods import week2_nonadaptive
from model import Video
from metrics import mean_average_precision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def w2_map_alpha(alpha):
    video = Video("../datasets/AICity_data/train/S03/c010/vdo.avi")
    frames = []
    for frame in week2_nonadaptive(video, alpha):
        # iou = frame.get_detection_iou(ignore_classes=True)
        # result = frame.to_result(ignore_classes=True)
        frames.append(frame)
    mAP = mean_average_precision(frames)

    print('alpha', alpha, 'mAP', mAP)

    return mAP


def main():
    mAP_list = []
    alpha_values = np.linspace(1.5, 3, 20)
    for alpha in tqdm(alpha_values, file=sys.stdout, desc='Global loop....'):
        mAP_list.append(w2_map_alpha(alpha))

    fig, ax = plt.figure
    ax.plot(alpha_values, mAP_list)
    ax.set(xlabel='alpha value', ylabel='mean average precision')
    plt.show()


if __name__ == '__main__':
    main()
