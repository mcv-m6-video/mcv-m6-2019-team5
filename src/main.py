import argparse

from methods import week2_nonadaptive, week2_adaptive
from model import Video

method_refs = {
    'w2_adaptive': week2_adaptive,
    'w2_nonadaptive': week2_nonadaptive
}


def main():
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')

    parser.add_argument('method', help='Method to use', choices=method_refs.keys())

    args = parser.parse_args()

    method = method_refs.get(args.method)

    video = Video("../datasets/AICity_data/train/S03/c010/vdo.avi")

    results = method(video)


if __name__ == '__main__':
    main()
