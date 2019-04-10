from typing import List

from model import Detection


def read_detections(path: str) -> List[List[Detection]]:
    # [frame, -1, left, top, width, height, conf, -1, -1, -1]
    frame_detections = []
    with open(path) as f:
        for line in f.readlines():
            parts = line.split(',')

            frame_id = int(parts[0])
            while frame_id > len(frame_detections):
                frame_detections.append([])

            tl_x = int(float(parts[2]))
            tl_y = int(float(parts[3]))
            width = int(float(parts[4]))
            height = int(float(parts[5]))
            confidence = float(parts[6])

            frame_detections[-1].append(
                Detection(int(parts[1]), 'car', (tl_x, tl_y), width, height, confidence=confidence))

    return frame_detections
