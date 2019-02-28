import cv2
import matplotlib.pyplot as plt
import numpy as np


class Picture:
    image_cached: np.ndarray
    name: str
    parent_dir: str
    id: int

    def __init__(self, parent_dir: str, name: str):
        self.name = name
        self.parent_dir = parent_dir
        self.image_cached = None
        self.id = int(self.name[4:-4])

    def get_image(self) -> np.array:
        if self.image_cached is None:
            self.image_cached = cv2.imread(self.parent_dir + '/' + self.name, cv2.IMREAD_COLOR)

        return self.image_cached

    def show(self):
        plt.figure()
        rgb = cv2.cvtColor(self.get_image(), cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
        plt.show()

    def show_hsv(self):
        plt.figure()
        rgb = cv2.cvtColor(self.get_image(), cv2.COLOR_BGR2HSV)
        plt.imshow(rgb[:, :, 0], 'gray')
        plt.show()

    def show_gray(self):
        plt.figure()
        gray = cv2.cvtColor(self.get_image(), cv2.COLOR_BGR2GRAY)
        plt.imshow(gray, 'gray')
        plt.show()

    def get_trimmed_name(self):
        return self.name.replace('.jpg', '')
