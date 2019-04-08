import argparse

from functional import seq

from model import Sequence
from tracking import KalmanTracking


def main():
    parser = argparse.ArgumentParser(description='Week 5 M6')

    parser.add_argument('-d', '--debug', action='store_true', help='Show debug plots')
    args = parser.parse_args()

    kalman = KalmanTracking()
    for video in Sequence("../datasets/AICity_data/train/S03").get_videos():
        for frame in video.get_frames():
            kalman(frame, args.debug)
            print(seq(frame.detections).map(lambda d: d.id).to_list())


if __name__ == '__main__':
    main()
