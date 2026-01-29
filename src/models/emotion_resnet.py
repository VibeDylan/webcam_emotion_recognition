"""ResNet18-based model for 7-class emotion recognition from 48x48 grayscale faces."""
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class EmotionResNet(nn.Module):
    """ResNet18 with first conv adapted to 1 channel; optional ImageNet pretrained weights."""

    def __init__(self, num_classes=7, pretrained=True):
        """Build ResNet18 for 1-channel input and num_classes. pretrained: use ImageNet weights for backbone."""
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
        """Forward pass. x: (B, 1, 48, 48). Returns logits (B, num_classes)."""
        return self.resnet(x)
