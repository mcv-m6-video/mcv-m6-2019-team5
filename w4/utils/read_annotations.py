import xml.etree.ElementTree as ET
from typing import List

from model import Detection
from .memory import memory


@memory.cache()
def read_annotations(file_path: str, frames: int = 2140) -> List[List[Detection]]:
    frames_detections = []

    root = ET.parse(file_path).getroot()
    tracks = root.findall('track')

    for i in range(frames + 1):
        frame_detections = []
        for track in tracks:
            id_value = int(track.attrib["id"])
            label = track.attrib["label"]
            if label == 'bike':
                label = 'bicycle'
            box = track.find('box[@frame="{}"]'.format(i))
            if box is not None:
                xtl = int(float((box.attrib["xtl"])))
                ytl = int(float((box.attrib["ytl"])))
                xbr = int(float((box.attrib["xbr"])))
                ybr = int(float((box.attrib["ybr"])))

                frame_detections.append(Detection(id_value, label, (xtl, ytl), xbr - xtl + 1, ybr - ytl + 1))

        frames_detections.append(frame_detections)

    return frames_detections
