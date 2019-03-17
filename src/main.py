import argparse

from model import Video

method_refs = {
}


def main():
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')

    parser.add_argument('method', help='Method to use', choices=method_refs.keys())
    parser.add_argument('--debug', action='store_true', help='Show debug plots')

    args = parser.parse_args()

    method = method_refs.get(args.method)

    video = Video("../datasets/AICity_data/train/S03/c010/frames")


if __name__ == '__main__':
    main()
