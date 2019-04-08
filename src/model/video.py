import os
from typing import Iterator

from PIL import Image
from torch.utils.data import Dataset
import cv2

from utils import read_detections


class Video(Dataset):

    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_frames(self):
        video = cv2.VideoCapture(os.path.join(self.video_path, "vdo.avi"))
        full_detections = read_detections(os.path.join(self.video_path, "det/"))
        count = 0
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                yield full_detections[count], frame,
                count +=1
            else:
                video.release()
