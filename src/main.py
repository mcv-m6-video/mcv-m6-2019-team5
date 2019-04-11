import argparse
import configparser
import sys

import cv2
import numpy as np
from matplotlib import patches, colors
from tqdm import tqdm

from metrics.mot import Mot
from model import Sequence, SiameseDB
from tracking import KalmanTracking, OverlapTracking, OpticalFlowTracking
from tracking.remove_parked_cars import RemoveParkedCars
from utils import write_detections
import matplotlib.pyplot as plt

methods = {
    'kalman': KalmanTracking,
    'overlap': OverlapTracking,
    'optical_flow': OpticalFlowTracking
}


def main():
    parser = argparse.ArgumentParser(description='Week 5 M6')
    parser.add_argument('tracking_type', type=str, choices=('single', 'multiple'), default='multiple',
                        nargs='?')
    parser.add_argument('sequence', type=str,
                        choices=('train_seq1', 'train_seq3', 'train_seq4', 'test_seq2', 'test_seq5'),
                        default='train_seq3',
                        nargs='?')
    parser.add_argument('tracking_method', type=str, choices=methods.keys(), default='kalman', nargs='?')
    parser.add_argument('-d', '--debug', action='store_true', help='Show debug plots')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('../config/config.ini')
    siamese = None
    if 'multiple' in args.tracking_type:
        siamese = SiameseDB(int(config.get(args.sequence, 'dimensions')), config.get(args.sequence, 'weights_path'))

    mot = Mot()
    i = 0

    idf1_list = []
    test_det_list = []

    for video in Sequence(config.get(args.sequence, 'sequence_path')).get_videos():
        remove_parked_cars = RemoveParkedCars()
        method = methods.get(args.tracking_method)()

        for frame in tqdm(video.get_frames(), file=sys.stdout, desc='Video {}'.format(video.get_name()),
                          total=len(video)):
            method(frame, siamese, args.debug)
            mot_detections = remove_parked_cars(frame)
            if frame.id > 300:
                break

            if frame.id == 250 or frame.id == 190 or frame.id == 200 or frame.id == 210 or frame.id == 220 or frame.id == 230:
                viridis = colors.ListedColormap(np.random.rand(256, 3))
                new_color = viridis(np.linspace(0, 1, 256))
                plt.imshow(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
                plt.axis('off')

                for det in mot_detections:
                    if det.id != -1:
                        rect = patches.Rectangle((det.top_left[0], det.top_left[1]), det.width, det.height,
                                                 linewidth=2, edgecolor=new_color[det.id, :], facecolor='none')
                        plt.gca().add_patch(rect)
                        plt.text(det.top_left[0] - 0, det.top_left[1] - 50, s='{}'.format(61),
                                 color='white', verticalalignment='top',
                                 bbox={'color': 'blue', 'pad': 0})
                        plt.gca().add_patch(rect)
                plt.imshow(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
                plt.close()

            if 'multiple' in args.tracking_type:
                siamese.process_frame(frame, mot_detections)

            if 'train' in args.sequence:
                mot.update(mot_detections, frame.ground_truth)
            elif 'test' in args.sequence:
                test_det_list.append(mot_detections)
            # print(seq(frame.detections).map(lambda d: d.id).to_list())
        if 'train' in args.sequence:
            if 'multiple' in args.tracking_type:
                siamese.update_db()
            elif 'single' in args.tracking_type:
                idf1, idp, idr, precision, recall = mot.get_metrics()
                idf1_list.append(idf1)
                print('Video {}:'.format(video.get_name()), idf1, idp, idr, precision, recall)
                mot = Mot()

        i += 1

    if 'train' in args.sequence:
        if args.tracking_type == 'multiple':
            idf1, idp, idr, precision, recall = mot.get_metrics()
            print('Metrics:', idf1, idp, idr, precision, recall)
        elif args.tracking_type == 'single':
            print('Mean idf1:', np.mean(idf1_list))
    elif 'test' in args.sequence:
        write_detections('../{}_{}.txt'.format(args.method, args.sequence), test_det_list)
    print(args.tracking_type, args.sequence, args.tracking_method)


if __name__ == '__main__':
    main()
