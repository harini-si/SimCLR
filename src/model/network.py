import torch
class ResNet(torch.nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def forward(self, x):
        return x