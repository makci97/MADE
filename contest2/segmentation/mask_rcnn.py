from torchvision import models
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_detector_model():
    model = models.detection.maskrcnn_resnet50_fpn(
        pretrained=True,
        pretrained_backbone=True,
        progress=True,
        num_classes=91,
    )

    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.box_predictor = box_predictor

    mask_predictor = MaskRCNNPredictor(256, 256, num_classes)
    model.roi_heads.mask_predictor = mask_predictor

    # Заморозим все слои кроме последних

    for param in model.parameters():
        param.requires_grad = False

    for param in model.backbone.fpn.parameters():
        param.requires_grad = True

    for param in model.rpn.parameters():
        param.requires_grad = True

    for param in model.roi_heads.parameters():
        param.requires_grad = True

    return model
