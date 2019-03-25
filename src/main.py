# Set up custom environment before nearly anything else is imported
# noinspection PyUnresolvedReferences
import argparse

from methods import fine_tune_yolo, off_the_shelf_yolo, off_the_shelf_ssd, siamese_train
from operations import SiameseTracking, OverlapTracking, KalmanTracking

method_refs = {
    'fine_tune_yolo': fine_tune_yolo,
    'off_the_shelf_yolo': off_the_shelf_yolo,
    'off_the_shelf_ssd': off_the_shelf_ssd,
    'siamese_train': siamese_train
}

tracking_refs = {
    'siamese': SiameseTracking(),
    'overlap': OverlapTracking(),
    'kalman': KalmanTracking()
}


def main():
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')

    parser.add_argument('method', help='Method to use', choices=method_refs.keys())
    parser.add_argument('tracking', help='Tracking method to use', choices=tracking_refs.keys(), default=None)
    parser.add_argument('-d', '--debug', action='store_true', help='Show debug plots')
    parser.add_argument('-e', '--epochs', type=int, help='Number of train epochs', default=25)

    args = parser.parse_args()

    tracking = None
    if args.tracking is not None:
        tracking = tracking_refs.get(args.tracking)

    method = method_refs.get(args.method)

    method(args.epochs, args.debug, tracking=tracking)


if __name__ == '__main__':
    main()
