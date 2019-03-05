import argparse
from multiprocessing import Queue, Process

from model import Video


def read_video_process(frame_queue: Queue):
    video = Video("../datasets/AICity_data/train/S03/c010/vdo.avi")

    for data in video.get_frames():
        frame_queue.put(data)


def main():
    """
    Read the video in a different process and buffer 10 frames
    """
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')

    parser.add_argument('methods', help='Method list separated by ;')

    args = parser.parse_args()

    method_refs = {
    }

    queue = Queue(maxsize=10)
    p = Process(target=read_video_process, args=(queue,))
    p.start()

    while p.is_alive() or not queue.empty():
        im, frame = queue.get()
        print(frame.id)

    p.join()


if __name__ == '__main__':
    main()
