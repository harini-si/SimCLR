import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import cv2


class SimCLRDataset(Dataset):
    def __init__(self, name, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.name = name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        if self.transform:
            if self.name == "stl10":
                img1 = self.transform(sample[0])
                img2 = self.transform(sample[0])
            else:

                img1 = self.transform(sample["image"])
                img2 = self.transform(sample["image"])

                if img1.shape[0] == 1:
                    img1 = img1.repeat(3, 1, 1)
                    img2 = img2.repeat(3, 1, 1)
        assert img1.shape == img2.shape

        return [img1, img2]
