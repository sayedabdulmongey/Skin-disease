from torchvision import datasets, transforms, models
import torch.nn as nn


class CustomEfficientNet(nn.Module):

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        self.model = models.efficientnet_b0(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
