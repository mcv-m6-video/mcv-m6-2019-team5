import argparse
import configparser

from functional import seq

from model import Sequence
from tracking import KalmanTracking


def main():
    parser = argparse.ArgumentParser(description='Week 5 M6')
    parser.add_argument('sequence', type=str, choices=('train_seq1', 'train_seq3', 'train_seq4'), default='train_seq3')
    parser.add_argument('tracking_method', type=str, choices=('kalman', 'overlap', 'optical_flow'), default='kalman')
    parser.add_argument('-d', '--debug', action='store_true', help='Show debug plots')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('configuration/config.ini')

    if args.tracking_method is 'kalman':
        kalman = KalmanTracking()
    for video in Sequence(config.get(args.sequence, 'sequence_path')).get_videos():
        for frame in video.get_frames():
            if args.tracking_method is 'kalman':
                kalman(frame, args.debug)
            print(seq(frame.detections).map(lambda d: d.id).to_list())


if __name__ == '__main__':
    main()
