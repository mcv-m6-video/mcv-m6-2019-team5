from utils import read_detections
import cv2
import os


def create_siamese_dataset(sequence, camera):
    AICity_dataset_path = '../datasets/AICity_data/train/' + sequence + '/' + camera + '/'
    siamese_dataset_path = '../datasets/full_siamese_data/'

    annotation_file = 'gt/gt.txt'
    gt_detections = read_detections(AICity_dataset_path + annotation_file)

    frames = len(gt_detections)

    for frame in range(frames):
        try:
            frame_name = "frame{:04d}.jpg".format(frame)
            img = cv2.imread(AICity_dataset_path + 'frames/' + frame_name)
            print(AICity_dataset_path + 'frames/' + frame_name)

            for detection in gt_detections[frame]:
                xtl, ytl = detection.top_left
                w = detection.width
                h = detection.height

                class_path = sequence + '-' + str(detection.id) + '/'
                if not os.path.exists(siamese_dataset_path + class_path):
                    os.mkdir(siamese_dataset_path + class_path)
                cropped_img = img[ytl:ytl + h, xtl:xtl + w]
                cv2.imwrite(siamese_dataset_path + class_path + frame_name, cropped_img)
        except:
            import code
            code.interact(local=dict(globals(), **locals()))


def main():
    S01_paths = os.listdir('../datasets/AICity_data/train/S01')
    S03_paths = os.listdir('../datasets/AICity_data/train/S03')
    S04_paths = os.listdir('../datasets/AICity_data/train/S04')
    sequences_path = ['S01'] * len(S01_paths) + ['S03'] * len(S03_paths) + ['S04'] * len(S04_paths)

    for path, sequence in zip(S01_paths + S03_paths + S04_paths, sequences_path):
        create_siamese_dataset(sequence, path)


if __name__ == '__main__':
    main()
