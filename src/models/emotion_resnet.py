import torch 
import torch.nn as nn 
import torchvision.models as models

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        
        resnet = models.resnet18(pretrained=pretrained)
        
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
