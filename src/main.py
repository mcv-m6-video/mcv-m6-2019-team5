import argparse

from methods import week1
from model import Video


def main():
    """
    Read the video in a different process and buffer 10 frames
    """
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')

    parser.add_argument('methods', help='Method list separated by ;')

    args = parser.parse_args()

    method_refs = {
    }

    video = Video("../datasets/AICity_data/train/S03/c010/vdo.avi")

    results = week1(video)


if __name__ == '__main__':
    main()
