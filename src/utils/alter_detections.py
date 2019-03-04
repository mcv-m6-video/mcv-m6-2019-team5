from typing import List
from model import Detection
import random


def alter_detections(detections: List[Detection]) -> List[Detection]:
    prob_not = 0.05
    translation = 30
    scale = [0.5, 1]
    prob_fp = 0.1

    frame_detections = []

    for d in detections:
        if random.uniform(0, 1) < prob_not:
            continue

        tl_x, tl_y = d.top_left
        tl_x += random.uniform(0, 1) * translation
        tl_y += random.uniform(0, 1) * translation
        width = d.width * random.uniform(scale[0], scale[1])
        height = d.height * random.uniform(scale[0], scale[1])

        frame_detections.append(Detection(d.id, d.label, (tl_x, tl_y), width, height))
        while random.uniform(0, 1) < prob_fp:
            frame_detections.append(Detection('', 'car', (random.uniform(0, 100), random.uniform(0, 900)),
                                              random.uniform(50, 150), random.uniform(50, 150)))

    return frame_detections
