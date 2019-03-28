from PIL import Image

from model import Frame, Rectangle
import numpy as np


def crop_image(im: Image, rectangle: Rectangle):
    return im.crop((rectangle.top_left[0], rectangle.top_left[1],
                    rectangle.top_left[0] + rectangle.width, rectangle.top_left[1] + rectangle.height))
