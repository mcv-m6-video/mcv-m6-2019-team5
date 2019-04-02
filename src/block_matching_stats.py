import itertools
import sys
import time
import cv2
from tqdm import tqdm

from metrics import pepn, msen
from optical_flow import BlockMatching

from utils import read_optical_flow


def main():
    im1 = cv2.imread('../datasets/optical_flow/img/000045_10.png')
    im2 = cv2.imread('../datasets/optical_flow/img/000045_11.png')
    gt = read_optical_flow('../datasets/optical_flow/gt/000045_10.png')

    block_sizes = [5, 7, 13, 17, 19, 23]
    window_sizes = [15, 25, 35, 51]
    costs = ['SSD', 'SAD']

    results = []

    for block_size, win_size, cost in tqdm(itertools.product(block_sizes, window_sizes, costs), file=sys.stdout,
                                           total=6 * 6 * 2):
        if win_size < block_size // 2:
            continue
        bm = BlockMatching(block_size=block_size, window_size=win_size, criteria=cost)
        t0 = time.time()
        flow = bm(im1, im2)
        t1 = time.time()

        msen_val = msen(flow, gt)
        pepn_val = pepn(flow, gt)
        time_val = t1 - t0

        results.append([block_size, win_size, cost, msen_val, pepn_val, time_val])

        print(results)


if __name__ == '__main__':
    main()
