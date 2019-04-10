import argparse
import configparser
import sys

from tqdm import tqdm

from metrics.mot import Mot
from model import Sequence, SiameseDB
from tracking import KalmanTracking, OverlapTracking, OpticalFlowTracking
from tracking.remove_parked_cars import RemoveParkedCars

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
    if 'multiple' in args.tracking_type:
        siamese = SiameseDB(int(config.get(args.sequence, 'dimensions')), config.get(args.sequence, 'weights_path'))
    method = methods.get(args.tracking_method)

    mot = Mot()

    for video in Sequence(config.get(args.sequence, 'sequence_path')).get_videos():
        remove_parked_cars = RemoveParkedCars()

        for frame in tqdm(video.get_frames(), file=sys.stdout, desc='Video {}'.format(video.get_name()),
                          total=len(video)):
            method(frame, siamese, args.debug)
            if 'multiple' in args.tracking_type:
                siamese.process_frame(frame)

            mot_detections = remove_parked_cars(frame)

            mot.update(mot_detections, frame.ground_truth)
            # print(seq(frame.detections).map(lambda d: d.id).to_list())
        if 'multiple' in args.tracking_type:
            siamese.update_db()
        elif 'single' in args.tracking_type:
            idf1 = mot.get_idf1()
            print('Video {} IDF1: {}'.format(video.get_name(), idf1))
            mot = Mot()

    if 'multiple' in args.tracking_type:
        idf1 = mot.get_idf1()
        print('IDF1: {}'.format(idf1))


if __name__ == '__main__':
    main()
