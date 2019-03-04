import xml.etree.ElementTree as ET
from typing import Iterator, Tuple

import cv2
import numpy as np

from model import Frame, Detection


class Video:
    video_path: str
    annotation_path: str
    car_only: bool

    def __init__(self, video_path: str, annotation_path: str, car_only: bool = False):
        self.video_path = video_path
        self.annotation_path = annotation_path
        self.car_only = car_only

    def get_frames(self) -> Iterator[Tuple[np.ndarray, Frame]]:
        cap = cv2.VideoCapture(self.video_path)
        root = ET.parse(self.annotation_path).getroot()

        num = 0
        while cap.isOpened():
            valid, image = cap.read()
            if not valid:
                break

            ground_truths = []
            for track in root.findall('track'):
                id = track.attrib['id']
                label = track.attrib['label']
                box = track.find("box[@frame='{0}']".format(str(num)))

                if self.car_only and label != 'car':
                    continue

                if box is not None:
                    xtl = int(float(box.attrib['xtl']))
                    ytl = int(float(box.attrib['ytl']))
                    xbr = int(float(box.attrib['xbr']))
                    ybr = int(float(box.attrib['ybr']))

                    ground_truths.append(Detection(id, label, (xtl, ytl), xbr - xtl + 1, ybr - ytl + 1))

            yield image, Frame(num, ground_truths)

            num += 1

        cap.release()
