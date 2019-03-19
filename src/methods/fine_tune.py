from torchvision import transforms

from model import Video
from utils import utils


def fine_tune(debug=False):
    video = Video("../datasets/AICity_data/train/S03/c010/frames")
    trans = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor()
    ])
    classes = utils.load_classes('../config/coco.names')
