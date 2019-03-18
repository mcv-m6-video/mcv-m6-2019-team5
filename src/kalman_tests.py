# Set up custom environment before nearly anything else is imported
# noinspection PyUnresolvedReferences
import argparse

from functional import seq

from model import Video, Frame
from operations import KalmanTracking
from utils import read_annotations, read_detections


def main():
    start_frame = 1440
    end_frame = 1789

    gt = read_annotations('../annotations', start_frame, end_frame)

    alg = 'mask_rcnn'

    detections = read_detections('../datasets/AICity_data/train/S03/c010/det/det_{0}.txt'.format(alg))

    kalman = KalmanTracking()
    for i in range(start_frame, end_frame):
        f = Frame(i)
        f.detections = detections[i]
        f.ground_truth = gt[i - start_frame]
        kalman(f)
        print(seq(f.detections).map(lambda d: d.id).to_list())


if __name__ == '__main__':
    main()
