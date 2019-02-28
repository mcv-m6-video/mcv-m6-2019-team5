# from model import Frame
import cv2
from xml.dom.minidom import parse, parseString


class Video:
    # frames: List[Frame]

    def __init__(self, video_path: str, annotation_path: str):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()

            # TODO create frame list

            cv2.imshow('frame', frame)

        cap.release()
        cv2.destroyAllWindows()
        # TODO


video = Video("../../datasets/AICity_data/train/S03/c010/vdo.avi", "")
