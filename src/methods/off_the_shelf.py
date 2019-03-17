import torch
from faster_rcnn.model.faster_rcnn.resnet import resnet

from model import Video


def off_the_shelf(video: Video):
    config_file = "configs/e2e_faster_rcnn_R_50_FPN_1x.yaml"

    model = resnet([1, 2, 3], 152, pretrained=True).cuda()

    print(model)
