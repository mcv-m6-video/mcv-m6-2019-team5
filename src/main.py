import argparse

from tracking import track_detections


def main():
    parser = argparse.ArgumentParser(description='Week 5 M6')

    parser.add_argument('-d', '--debug', action='store_true', help='Show debug plots')
    args = parser.parse_args()
    track_detections(args.debug)


if __name__ == '__main__':
    main()
