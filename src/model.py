import torchvision
import torch

def get_model_instance_segmentation(num_classes):
 # Load a pre-trained Faster R-CNN model
 model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
 in_features = model.roi_heads.box_predictor.cls_score.in_features
 model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
 return model
