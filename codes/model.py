import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CamelyonClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        """
        ResNet18-based classifier for CAMELYON17.
        Args:
            num_classes (int): Number of output classes (1 for binary classification with BCEWithLogitsLoss).
            pretrained (bool): Whether to use ImageNet pre-trained weights. Default False (train from scratch).
        """
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        
        # Modify the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)

def get_model(device='cuda'):
    model = CamelyonClassifier(num_classes=1, pretrained=False)
    model = model.to(device)
    return model
