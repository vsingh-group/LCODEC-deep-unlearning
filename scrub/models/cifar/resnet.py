import torch

def resnet18(**kwargs):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=10)
    return model

def resnet50(**kwargs):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False, num_classes=10)
    return model
