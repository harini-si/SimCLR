import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, out_dim, base_model="resnet18"):
        super(ResNet, self).__init__()

        self.backbone = self._get_basemodel(base_model, out_dim)

        # Adjusting the fully connected layer of the backbone
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc
        )

    def _get_basemodel(self, model_name, out_dim):
        model = models.resnet18(pretrained=False, num_classes=out_dim)
        return model

    def forward(self, x):
        return self.backbone(x)
