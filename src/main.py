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

from model import Data, Picture
from model.rectangle import Rectangle
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')

    parser.add_argument('methods', help='Method list separated by ;')

    args = parser.parse_args()

    method_refs = {
    }


if __name__ == '__main__':
    main()
