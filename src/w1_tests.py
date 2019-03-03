from functional import seq

from model import Video
from utils import read_detections


def main():
    video = Video("../datasets/AICity_data/train/S03/c010/vdo.avi",
                  "../datasets/AICity_data/train/S03/c010/Anotation_40secs_AICITY_S03_C010.xml")

    detections = read_detections('../datasets/AICity_data/train/S03/c010/det/det_ssd512.txt')

    for im, f in seq(video.get_frames()).take(40):
        f.detections = detections[f.id]
        print(f.to_result())


if __name__ == '__main__':
    main()
