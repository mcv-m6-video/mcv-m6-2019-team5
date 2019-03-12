import argparse

from methods import week2_nonadaptive, week2_adaptive, week2_soa, week2_soa_mod, week2_nonadaptive_hsv, \
    week2_adaptive_hsv
from metrics import iou_over_time, mean_average_precision
from model import Video

method_refs = {
    'w2_adaptive': week2_adaptive,
    'w2_nonadaptive': week2_nonadaptive,
    'w2_soa': week2_soa,
    'w2_nonadaptive_hsv': week2_nonadaptive_hsv,
    'w2_adaptive_hsv': week2_adaptive_hsv,
    'w2_soa_mod': week2_soa_mod
}


def main():
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')

    parser.add_argument('method', help='Method to use', choices=method_refs.keys())
    parser.add_argument('--debug', action='store_true', help='Show debug plots')

    args = parser.parse_args()

    method = method_refs.get(args.method)

    video = Video("../datasets/AICity_data/train/S03/c010/vdo.avi")

    frames = []
    for frame in method(video, **{'debug': args.debug}):
        frames.append(frame)
        iou = frame.get_detection_iou(ignore_classes=True)
        print(iou)
    iou_over_time(frames, ignore_classes=True)
    print('mAP:', mean_average_precision(frames, ignore_classes=True))


if __name__ == '__main__':
    main()
