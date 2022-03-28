import torch

def resnext(**kwargs):
    """Constructs a ResNeXt.
    """
    #model = CifarResNeXt(**kwargs)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)#, num_classes=10)
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=False)#, num_classes=10)

    return model
