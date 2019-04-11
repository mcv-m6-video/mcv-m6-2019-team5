from typing import List

from model import Detection


def write_detections(path: str, det: List[List[Detection]]):
    file = open(path, "w")
    frame_id = 0
    for det1 in det:
        frame_id += 1
        for det2 in det1:
            out = str(frame_id) + "," + str(det2.id) + "," + str(det2.top_left[0]) + "," + str(
                det2.top_left[1]) + "," + str(det2.width) + "," + str(det2.height) + "," + str(
                det2.confidence) + "," + str(-1) + "," + str(-1) + "," + str(-1)
            file.write(out)
    file.close()
