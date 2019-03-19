import os
import xml.etree.ElementTree as ET
from typing import List

from model import Detection


def read_annotations(file_path: str) -> List[List[Detection]]:
    frames_detections = []

    root = ET.parse(file_path).getroot()

    frame_detections = []

    for obj in root.findall('object'):
        box = obj.find('bndbox')

        label = obj.find('name').text
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)

        frame_detections.append(Detection('', label, (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1))

    frames_detections.append(frame_detections)


return frames_detections
