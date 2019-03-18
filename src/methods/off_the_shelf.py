from model import Video
from yolo.models import Darknet
import os
import wget


def off_the_shelf(video: Video):
    model = Darknet('../config/yolov3.cfg')
    if not os.path.exists('../.cache/yolov3.weights'):
        if not os.path.exists('../.cache'):
            os.makedirs('../.cache/')
        print('Downloading weights...')
        wget.download('https://pjreddie.com/media/files/yolov3.weights', out='../.cache/')
        print('Weights downloaded')

    model.load_weights('../.cache/yolov3.weights')

    print(model)
