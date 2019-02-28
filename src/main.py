import argparse
import fnmatch
import os
import pickle
import sys
from typing import List
from itertools import product

import ml_metrics as metrics
import pandas
from functional import seq

from methods import AbstractMethod, w5, w5_no_frame, w5_no_frame_no_text, ycbcr_16_hellinger
from model import Data, Picture
from model.rectangle import Rectangle
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')

    parser.add_argument('methods', help='Method list separated by ;')

    args = parser.parse_args()

    method_refs = {
        'w5': w5,
        'w5_no_frame': w5_no_frame,
        'w5_no_frame_no_text': w5_no_frame_no_text,
        'ycbcr_16_hellinger': ycbcr_16_hellinger
    }


if __name__ == '__main__':
    main()
