import motmetrics as mm


class Mot:

    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, detections, gts):
        detection_bboxes = [(detection.top_left[0], detection.top_left[1], detection.width, detection.height) for
                            detection in detections]
        gt_bboxes = [(gt.top_left[0], gt.top_left[1], gt.width, gt.height) for gt in gts]

        gt_ids = [gt.id for gt in gts]
        detection_ids = [detection.id for detection in detections]

        distances_matrix = mm.distances.iou_matrix(gt_bboxes, detection_bboxes, max_iou=1.)

        self.acc.update(gt_ids, detection_ids, distances_matrix)

    def get_idf1(self):
        mh = mm.metrics.create()
        idf1 = mh.compute(self.acc, metrics=['idf1'], name='acc')
        return idf1

    def get_events(self):
        return self.acc.mot_events

    def create_events(self, frame_id, det):
        indices = []
        events = []
        for event in len(det):
            indices.append((frame_id, event))
            events.append(('MATCH',))
        # self.acc.new_event_dataframe_with_data(indices, )
