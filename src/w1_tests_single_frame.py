import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from model import Frame, Detection
from utils import alter_detections

amount_frames = 40
make_video = True


def main():
    im_1440 = cv2.imread("../datasets/AICity_data_S03_c010_1440/frame_1440.jpeg")
    top_left = [995, 410]
    width = 1241 - 995
    height = 605 - 410
    ground_truth = [Detection('', 'car', top_left, width, height)]

    """
        DETECTIONS FROM ALTERED GROUND TRUTH 
    """
    frame = Frame(0, ground_truth)
    frame.detections = alter_detections(ground_truth)

    plot_frame(im_1440, frame)
    iou = frame.get_detection_iou()
    iou_mean = frame.get_detection_iou_mean()
    print("IOU: ", iou, "IOU mean", iou_mean)


def plot_frame(im, frame):
    im1 = im.copy()

    for d in frame.ground_truth:
        cv2.rectangle(im1, (int(d.top_left[0]), int(d.top_left[1])),
                      (int(d.get_bottom_right()[0]), int(d.get_bottom_right()[1])), (255, 0, 0), thickness=5)

    for d in frame.detections:
        cv2.rectangle(im1, (int(d.top_left[0]), int(d.top_left[1])),
                      (int(d.get_bottom_right()[0]), int(d.get_bottom_right()[1])), (0, 0, 255), thickness=5)

    plt.title('Ground truth')
    plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.legend([
        Line2D([0], [0], color=(0, 0, 1)),
        Line2D([0], [0], color=(1, 0, 0)),
    ], ['Ground truth', 'Detection'])

    plt.show()


if __name__ == '__main__':
    main()
