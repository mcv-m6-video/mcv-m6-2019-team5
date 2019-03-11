import sys
from typing import List

from model import Rectangle


def combine_overlapped_regions(regions: List[Rectangle]) -> List[Rectangle]:
    """
    Combines the possible overlapped regions into a set of non overlapped regions by using union.
    :param regions: the list of regions probably overlapped
    :return: a list of regions that are not overlapped
    """
    if len(regions) > 10000:
        print('Too many regions detected, returning []', file=sys.stderr)
        return []
    ret = list(regions)

    while combine(ret):
        pass

    return ret


def combine(ret: List[Rectangle]) -> bool:
    i = 0
    combined = False
    while i < len(ret):
        j = i + 1
        region = ret[i]
        while j < len(ret):
            if region.intersects(ret[j]):
                region = region.union(ret[j])
                ret[i] = region
                ret.remove(ret[j])
                j -= 1
                combined = True

            j += 1
        if combined:
            return True
        i += 1
    return False
