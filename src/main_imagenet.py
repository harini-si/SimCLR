import logging
import argparse
from datasets import load_dataset
import torch
from model.simclr import SimCLR
from model.network import ResNet
from data.dataset import SimCLRDataset
from torchvision import transforms

parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", type=str, help="The name of the dataset to use")
parser.add_argument("--batch_size", type=int, default=32, help="The batch size to use")
parser.add_argument(
    "--num_workers", type=int, default=4, help="The number of workers to use"
)
parser.add_argument(
    "--learning_rate", type=float, default=0.001, help="The learning rate to use"
)
parser.add_argument(
    "--epochs", type=int, default=2, help="The number of epochs to train for"
)
parser.add_argument(
    "--weight_decay", type=float, default=1e-6, help="The weight decay to use"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument(
    "--fp16-precision",
    action="store_true",
    help="Whether or not to use 16-bit precision GPU training.",
)
parser.add_argument("--device", default="cpu")
parser.add_argument(
    "--out_dim", default=64, type=int, help="feature dimension (default: 64)"
)
parser.add_argument(
    "--log-every-n-steps", default=100, type=int, help="Log every n steps"
)
parser.add_argument(
    "--temperature",
    default=0.07,
    type=float,
    help="softmax temperature (default: 0.07)",
)


args = parser.parse_args()
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading data")
    train_dataset = load_dataset("zh-plus/tiny-imagenet", split="train")
    # take only 10%
    train_dataset = train_dataset.select(range(0, len(train_dataset), 10))
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
    )

    train_dataset = SimCLRDataset(train_dataset, transform=data_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    model = ResNet(out_dim=64)

    optimizer = torch.optim.Adam(
        model.parameters(), args.learning_rate, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )

    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr.train(train_loader)
