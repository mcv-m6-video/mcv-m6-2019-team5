from utils import read_annotations
import cv2 as cv
import os


def create_siamese_dataset():
    AICity_dataset_path = '../datasets/AICity_data/train/S03/c010/'
    siamese_dataset_path = '../datasets/siamese_data/'
    annotation_file = 'm6-full_annotation.xml'
    frames = 2140
    gt_detections = read_annotations(AICity_dataset_path + annotation_file, frames)

    for frame in range(frames + 1):
        frame_name = "frame_{:04d}.jpg".format(frame + 1)
        img = cv.imread(AICity_dataset_path + 'frames/' + frame_name)

        for detection in gt_detections[frame]:
            xtl, ytl = detection.top_left
            w = detection.width
            h = detection.height

            class_path = str(detection.label) + '-' + str(detection.id) + '/'
            if not os.path.exists(siamese_dataset_path + class_path):
                os.mkdir(siamese_dataset_path + class_path)
            cropped_img = img[ytl:ytl + h, xtl:xtl + w]
            cv.imwrite(siamese_dataset_path + class_path + frame_name, cropped_img)


create_siamese_dataset()
