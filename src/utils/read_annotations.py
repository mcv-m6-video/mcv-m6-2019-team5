import os
import xml.etree.ElementTree as ET
from typing import List

from model import Detection


def read_annotations(root_directory: str, start: int, end: int) -> List[List[Detection]]:
    frames_detections: List[List[Detection]] = []

    for i in range(start, end + 1):
        frame_path = 'frame_{:04d}.xml'.format(i)
        root = ET.parse(os.path.join(root_directory, frame_path)).getroot()

        frame_detections: List[Detection] = []

        for obj in root.findall('object'):
            box = obj.find('bndbox')

            label = obj.find('name').text
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)

            frame_detections.append(Detection('', label, (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1))

    return frames_detections
