import torch
from torchvision import transforms

from model import Video
from yolo.models import Darknet


def off_the_shelf():
    video = Video("../datasets/AICity_data/train/S03/c010/frames", True, transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor()
    ]))

    model = Darknet('../config/yolov3.cfg')
    model.load_weights('../.weights/yolov3.weights')
    model = model.cuda()

    model.eval()
    with torch.no_grad():
        for im in video.get_frames():
            im = im.view((-1,) + im.size()).cuda()
            print(im.size())
            output = model.forward(im)
            print(output.size())
