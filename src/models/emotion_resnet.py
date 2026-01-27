import torch 
import torch.nn as nn 
import torchvision.models as models

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
