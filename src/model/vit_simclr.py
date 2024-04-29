import torch.nn as nn
import torchvision.models as models
import torch
from transformers import ViTModel, ViTConfig



    
class ViT(nn.Module):
    def __init__(self, out_dim, base_model="vit_base_patch16_224"):
        super(ViT, self).__init__()

        self.backbone = self._get_basemodel(base_model, out_dim)

        # Adjusting the fully connected layer of the backbone
        # dim_mlp = self.backbone.config.hidden_size
        # self.backbone.fc = nn.Sequential(
        #     nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc
        # )
        self.fc = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, self.backbone.config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone.config.hidden_size, out_dim)
        )

    def _get_basemodel(self, model_name, out_dim):
        config = ViTConfig(image_size=96)
        model = ViTModel(config)
        #load pre trained model
        # model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        return model

    def forward(self, x):
        x = self.backbone(x)['last_hidden_state'][:, 0]
        x = self.fc(x)
        return x
