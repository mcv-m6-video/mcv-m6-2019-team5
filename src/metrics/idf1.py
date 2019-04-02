import motmetrics as mm


def idf1(detections_list, gt_list, acc):
    detection_bboxes = [(detection.top_left[0], detection.top_left[1], detection.width, detection.height) for detection  in detections_list]
    gt_bboxes = [(gt.top_left[0], gt.top_left[1], gt.width, gt.height) for gt in gt_list]

    gt_ids = [gt.id for gt in gt_list]
    detection_ids = [detection.id for detection in detections_list]

    distances_matrix = mm.distances.iou_matrix(gt_bboxes, detection_bboxes, max_iou=1.)

    if idf1:
        acc.update(gt_ids, detection_ids, distances_matrix)
