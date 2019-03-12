from typing import Iterator

from model import Video, Frame
from operations.find_boxes import find_boxes
from operations.gaussian_model import get_background_model, gaussian_model_adaptive, PixelValue
from operations.morphological_operations import closing, opening
from utils import read_detections


def week2_adaptive_hsv(video: Video) -> Iterator[Frame]:
    model_mean, model_std = get_background_model(video, int(2141 * 0.25), total_frames=int(2141 * 0.25))

    ground_truth = read_detections('../datasets/AICity_data/train/S03/c010/gt/gt.txt')

    frame_id = int(2141 * 0.25)
    for mask in gaussian_model_adaptive(video, int(2141 * 0.25), model_mean, model_std, total_frames=int(2141 * 0.10),
                                        pixel_value=PixelValue.HSV):
        mask = opening(closing(mask, 15), 15)

        bbs = find_boxes(mask)

        frame = Frame(frame_id)
        frame.detections = bbs
        frame.ground_truth = ground_truth[frame_id]

        yield frame

        frame_id += 1
