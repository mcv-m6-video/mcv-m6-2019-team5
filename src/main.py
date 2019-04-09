import argparse
import configparser

from functional import seq

from model import Sequence, SiameseDB
from tracking import KalmanTracking, OverlapTracking


def main():
    parser = argparse.ArgumentParser(description='Week 5 M6')
    parser.add_argument('sequence', type=str, choices=('train_seq1', 'train_seq3', 'train_seq4'), default='train_seq3')
    parser.add_argument('tracking_method', type=str, choices=('kalman', 'overlap', 'optical_flow'), default='kalman')
    parser.add_argument('-d', '--debug', action='store_true', help='Show debug plots')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('configuration/config.ini')
    siamese = SiameseDB(config.get(args.sequence, 'dimensions'), config.get(args.sequence, 'weights_path'))
    if args.tracking_method is 'kalman':
        kalman = KalmanTracking()
    elif args.tracking_method is 'overlap':
        overlap = OverlapTracking()
    for video in Sequence(config.get(args.sequence, 'sequence_path')).get_videos():
        for frame in video.get_frames():
            if args.tracking_method is 'kalman':
                kalman(frame, siamese, args.debug)
            elif args.tracking_method is 'overlap':
                overlap(frame, siamese, args.debug)
            siamese.process_frame(frame)
            print(seq(frame.detections).map(lambda d: d.id).to_list())
        siamese.update_db()


if __name__ == '__main__':
    main()
