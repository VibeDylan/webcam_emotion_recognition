import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if pretrained:
            old_weights = resnet.conv1.weight.data
            new_weights = old_weights.mean(dim=1, keepdim=True)
            resnet.conv1.weight.data = new_weights
        
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, num_classes)
        
        self.resnet = resnet
    
    def forward(self, x):
        return self.resnet(x)
