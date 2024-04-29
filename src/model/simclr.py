import logging
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model.utils import accuracy, save_checkpoint, save_config_file

torch.manual_seed(0)


class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs["args"]
        self.model = kwargs["model"].to(self.args.device)
        self.optimizer = kwargs["optimizer"]
        self.scheduler = kwargs["scheduler"]
        self.writer = SummaryWriter()
        logging.basicConfig(
            filename=os.path.join(self.writer.log_dir, "training.log"),
            level=logging.DEBUG,
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features, temperature=0.07):
        features = F.normalize(features, dim=1)

        # Compute pairwise cosine similarity
        similarity_matrix = torch.matmul(features, features.T) / temperature

        # Exclude diagonal elements from consideration (similarity of each feature with itself)
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
        similarity_matrix = similarity_matrix - mask * 1e9

        # For each feature, compute log probability of all other features being positive pairs
        logits = similarity_matrix

        # Generate labels for each feature
        labels = torch.arange(logits.size(0), device=logits.device)

        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        n_iter = 0
        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):

                with autocast(enabled=False):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar("loss", loss, global_step=n_iter)
                    self.writer.add_scalar(
                        "learning_rate", self.scheduler.get_lr()[0], global_step=n_iter
                    )

                n_iter += 1

            if epoch_counter >= 10:
                self.scheduler.step()

            logging.info(f"Epoch: {epoch_counter}\tLoss: {loss}")

        logging.info("Training has finished.")
        checkpoint_name = "checkpoint_{:04d}.pth.tar".format(self.args.epochs)
        save_checkpoint(
            {
                "epoch": self.args.epochs,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            is_best=False,
            filename=os.path.join(self.writer.log_dir, checkpoint_name),
        )
        logging.info(
            f"Model checkpoint and metadata has been saved at {self.writer.log_dir}."
        )
