from model import Frame
from utils import read_detections
from metrics import mean_average_precision


def compute_map_gt_det(det_file, gt_file):
    det_list = read_detections(det_file)
    gt_list = read_detections(gt_file)

    frames = []

    for i in range(0, len(det_list)):
        frame = Frame(i)
        frame.detections = det_list[i]
        frame.ground_truth = gt_list[i]
        frames.append(frame)

    mAP = mean_average_precision(frames, ignore_classes=True)

    return mAP


def main():
    dataset_path = '../datasets/AICity_data/train/S03/c010/'
    gt_path = dataset_path + 'gt/gt.txt'
    rcnn_path = dataset_path + 'det/det_mask_rcnn.txt'
    ssd_path = dataset_path + 'det/det_ssd512.txt'
    yolo_path = dataset_path + 'det/det_yolo3.txt'

    mAP_ssd = compute_map_gt_det(ssd_path, gt_path)
    mAP_rcnn = compute_map_gt_det(rcnn_path, gt_path)
    mAP_yolo = compute_map_gt_det(yolo_path, gt_path)

    print(mAP_ssd, mAP_rcnn, mAP_yolo)


if __name__ == "__main__":
    main()
