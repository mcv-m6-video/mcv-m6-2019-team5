# from model import Frame
import cv2
import xml.etree.ElementTree as ET


class Video:
    # frames: List[Frame]

    def __init__(self, video_path: str, annotation_path: str):
        cap = cv2.VideoCapture(video_path)
        root = ET.parse(annotation_path).getroot()

        num = 0
        while cap.isOpened():
            _, frame = cap.read()

            for track in root.findall('track'):
                id = track.attrib['id']
                label = track.attrib['label']
                box = track.find("box[@frame='{0}']".format(str(num)))
                xtl = box.attrib['xtl']
                ytl = box.attrib['ytl']
                xbr = box.attrib['xbr']
                ybr = box.attrib['ybr']

            # TODO create frame list

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            num += 1
        cap.release()
        cv2.destroyAllWindows()
        # TODO


video = Video("../../datasets/AICity_data/train/S03/c010/vdo.avi",
              "../../datasets/AICity_data/train/S03/c010/Anotation_40secs_AICITY_S03_C010.xml")
