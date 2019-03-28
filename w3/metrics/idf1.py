from model import Frame
from typing import List


def idf1(frames: List[Frame]):
    nbox_gt = sum([frame.ground_truth.shape[0] for frame in frames])
    nbox_st = sum([frame.detections.shape[0] for frame in frames])

    fp, fn = [frame.to_result()[1::2] for frame in frames]

    idfp = 0
    idfn = 0

    matches_list = []  # implement function to get matched detections by object ID

    for matched in matches_list:
        idfp += fp[matched[0], matched[1]]
        idfn += fn[matched[0], matched[1]]
    idtp = nbox_gt - idfn
    assert idtp == nbox_st - idfp
    idf1 = 2 * idtp / (nbox_gt + nbox_st) * 100

    return idf1
