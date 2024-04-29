import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import STL10
from transformers import ViTModel, ViTConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

def get_stl10_data_loaders(download, shuffle=False, batch_size=256):
  train_dataset = datasets.STL10('./data', split='train', download=download,
                                  transform=transforms.ToTensor())

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  
  test_dataset = datasets.STL10('./data', split='test', download=download,
                                  transform=transforms.ToTensor())

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return train_loader, test_loader

class ViT(nn.Module):
    def __init__(self, out_dim, base_model="vit_base_patch16_224"):
        super(ViT, self).__init__()

        self.backbone = self._get_basemodel(base_model, out_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, self.backbone.config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone.config.hidden_size, out_dim)
        )

    def _get_basemodel(self, model_name, out_dim):
        config = ViTConfig(image_size=96)
        model = ViTModel(config)
        return model

    def forward(self, x):
        x = self.backbone(x)['last_hidden_state'][:, 0]
        x = self.fc(x)
        return x
    
model = ViT(out_dim=10).to(device)
print(model)
checkpoint = torch.load('runs/Apr25_02-03-54_dgx1.csis.bits-goa.ac.in/checkpoint_0100.pth.tar', map_location=device)
state_dict = checkpoint['state_dict']
# print(state_dict.keys())
state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
log = model.load_state_dict(state_dict, strict=False)
print(log.missing_keys)
assert log.missing_keys == ['fc.0.weight', 'fc.0.bias', 'fc.2.weight', 'fc.2.bias']
print("state loaded")
train_loader, test_loader = get_stl10_data_loaders(download=True)

# freeze all layers but the last fc
for name, param in model.named_parameters():
    # print(name)
      if 'backbone' in name:
    # if name not in ['mlp.1.weight', 'mlp.1.bias']:
        param.requires_grad = False

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
print(len(parameters))
assert len(parameters) == 4  # fc.weight, fc.bias


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss().to(device)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

epochs = 100
print("Starting training ...")
for epoch in range(epochs):
  top1_train_accuracy = 0
  for counter, (x_batch, y_batch) in enumerate(train_loader):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    top1 = accuracy(logits, y_batch, topk=(1,))
    top1_train_accuracy += top1[0]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  top1_train_accuracy /= (counter + 1)
  top1_accuracy = 0
  top5_accuracy = 0
  for counter, (x_batch, y_batch) in enumerate(test_loader):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)
  
    top1, top5 = accuracy(logits, y_batch, topk=(1,5))
    top1_accuracy += top1[0]
    top5_accuracy += top5[0]
  
  top1_accuracy /= (counter + 1)
  top5_accuracy /= (counter + 1)
  print(f"Epoch {epoch} Loss: {loss} Top1 Train accuracy: {top1_train_accuracy.item()} Top1 Test accuracy: {top1_accuracy.item()} Top5 test acc: {top5_accuracy.item()}")
