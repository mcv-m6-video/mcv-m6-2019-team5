import argparse
import configparser
import sys

from tqdm import tqdm

from metrics.mot import Mot
from model import Sequence, SiameseDB
from tracking import KalmanTracking, OverlapTracking, OpticalFlowTracking

methods = {
    'kalman': KalmanTracking(),
    'overlap': OverlapTracking(),
    'optical_flow': OpticalFlowTracking()
}


def main():
    parser = argparse.ArgumentParser(description='Week 5 M6')
    parser.add_argument('tracking_type', type=str, choices=('single', 'multiple'), default='multiple',
                        nargs='?')
    parser.add_argument('sequence', type=str, choices=('train_seq1', 'train_seq3', 'train_seq4'), default='train_seq3',
                        nargs='?')
    parser.add_argument('tracking_method', type=str, choices=methods.keys(), default='kalman', nargs='?')
    parser.add_argument('-d', '--debug', action='store_true', help='Show debug plots')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('../config/config.ini')
    siamese = None
    if args.tracking_type is 'multiple':
        siamese = SiameseDB(int(config.get(args.sequence, 'dimensions')), config.get(args.sequence, 'weights_path'))
    method = methods.get(args.tracking_method)

    mot = Mot()

    for video in Sequence(config.get(args.sequence, 'sequence_path')).get_videos():
        for frame in tqdm(video.get_frames(), file=sys.stdout, desc='Video {}'.format(video.get_name()),
                          total=len(video)):
            method(frame, siamese, args.debug)
            if args.tracking_type is 'multiple':
                siamese.process_frame(frame)

            mot.update(frame.detections, frame.ground_truth)
            # print(seq(frame.detections).map(lambda d: d.id).to_list())
        if args.tracking_type is 'multiple':
            siamese.update_db()
        elif args.tracking_type is 'single':
            idf1 = mot.get_idf1()
            print('Video {} IDF1: {}'.format(video.get_name(), idf1))
            mot = Mot()

    if args.tracking_type is 'multiple':
        idf1 = mot.get_idf1()
        print('IDF1: {}'.format(idf1))


if __name__ == '__main__':
    main()
