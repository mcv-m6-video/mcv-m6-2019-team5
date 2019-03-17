from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model


def fine_tune():
    config_file = "configs/e2e_faster_rcnn_R_50_FPN_1x.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)

    model = build_detection_model(cfg)
