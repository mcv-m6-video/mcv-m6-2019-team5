import sys

from joblib import Parallel, delayed

from methods import week2_nonadaptive
from model import Video
from metrics import mean_average_precision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def w2_map_alpha(alpha):
    video = Video("../datasets/AICity_data/train/S03/c010/frames")
    frames = []
    ious = []
    for im, mask, frame in week2_nonadaptive(video, alpha, disable_tqdm=True):
        frames.append(frame)
        ious.append(frame.get_detection_iou_mean(ignore_classes=True))

    mAP = mean_average_precision(frames)

    print('alpha', alpha, 'mAP', mAP, 'mean IoU', np.mean(ious))

    return np.mean(ious)


def main():
    alpha_values = np.linspace(1.5, 3, 20)
    mAP_list = Parallel(n_jobs=1)(delayed(w2_map_alpha)(alpha) for alpha in tqdm(alpha_values))
    # for alpha in tqdm(alpha_values, file=sys.stdout, desc='Global loop....'):
    #     mAP_list.append(w2_map_alpha(alpha))

    plt.figure()
    plt.plot(alpha_values, mAP_list)
    plt.xlabel(r'$\alpha$ threshold')
    plt.ylabel('IoU')
    plt.savefig('iou_alpha.png')
    plt.show()


if __name__ == '__main__':
    main()
