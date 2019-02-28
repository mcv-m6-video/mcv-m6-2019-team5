import fnmatch
import os
from typing import List

from model import Picture


class Data:
    pictures = List[Picture]

    def __init__(self, directory: str):
        self.pictures = []

        file_names = fnmatch.filter(os.listdir(directory), '*.jpg')
        for file_name in file_names:
            self.pictures.append(Picture(directory, file_name))
