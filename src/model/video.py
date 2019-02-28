from typing import List

from model import Frame, Detection
import cv2
import xml.etree.ElementTree as ET


class Video:
    frames: List[Frame]

    def __init__(self, video_path: str, annotation_path: str):
        self.frames = []

        cap = cv2.VideoCapture(video_path)
        root = ET.parse(annotation_path).getroot()

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
                if box is not None:
                    xtl = int(float(box.attrib['xtl']))
                    ytl = int(float(box.attrib['ytl']))
                    xbr = int(float(box.attrib['xbr']))
                    ybr = int(float(box.attrib['ybr']))

                    ground_truths.append(Detection(id, label, (xtl, ytl), xbr - xtl + 1, ybr - ytl + 1))

            self.frames.append(Frame(image, ground_truths))

            num += 1
            if num % 100 == 0:
                print(num)

        cap.release()
