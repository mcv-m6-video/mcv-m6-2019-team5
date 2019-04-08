import os
import cv2

from utils import read_detections


class Video():

    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_frames(self):
        video = cv2.VideoCapture(os.path.join(self.video_path, "vdo.avi"))
        full_detections = read_detections(os.path.join(self.video_path, "det/det_yolo3.txt"))
        full_ground_truth = read_detections(os.path.join(self.video_path, "gt/gt.txt"))

        count = 0
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                yield full_detections[count], full_ground_truth[count], frame
                count +=1
            else:
                video.release()
