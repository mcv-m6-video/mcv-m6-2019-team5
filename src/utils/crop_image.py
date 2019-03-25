from model import Frame, Rectangle
import numpy as np


def crop_image(im: np.ndarray, rectangle: Rectangle):
    top_left = rectangle.top_left
    return im[top_left[1]:top_left[1]+rectangle.height, top_left[0]:top_left[0]+rectangle.width]
