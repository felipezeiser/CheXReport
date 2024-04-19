import os
import random
from datetime import datetime
from typing import Dict, List, Union, Optional
from statistics import mean
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter

from nlg_metrics import Metrics
from utils.custom_types import ModelType, OptimType, DeviceType, DataIterType, SchedulerType
from utils.train_utils import seed_everything

class TrackMetrics:
    """ Helper class to track and log training and validation metrics. """

    def __init__(self) -> None:
        self.reset_running()
        self.metrics = self.init_metrics()

    def create_default_dict(self) -> Dict[str, defaultdict]:
        return {"train": defaultdict(list), "val": defaultdict(list)}

    def reset_running(self):
        """ Reset running metrics for a new accumulation period. """
        self.running = self.create_default_dict()

    def init_metrics(self) -> Dict[str, defaultdict]:
        """ Initialize a storage for metrics across the entire training process. """
        return self.create_default_dict()

    def update_running(self, metrics: Dict[str, float], phase: str) -> None:
        """ Update running metrics for the current batch. """
        for name, value in metrics.items():
            self.running[phase][name].append(value)

    def update(self, phase: str):
        """ Compute and store the mean of the running metrics and reset for next accumulation. """
        for name, values in self.running[phase].items():
            self.metrics[phase][name].append(mean(values))
        self.reset_running()

class Trainer:
    """ Main class for handling the training process. """

    def __init__(self, optims: List[OptimType], schedulers: List[SchedulerType], device: DeviceType,
                 epochs: int, val_interval: int, early_stop: int, lr_patience: int, embeddings_finetune: int,
                 grad_clip: float, lambda_c: float, checkpoints_path: str, pad_id: int, resume: Optional[str] = None) -> None:
        self.optims = optims
        self.schedulers = schedulers
        self.device = device
        self.epochs = epochs
        self.val_interval = val_interval
        self.early_stop = early_stop
        self.lr_patience = lr_patience
        self.embeddings_finetune = embeddings_finetune
        self.grad_clip = grad_clip
        self.lambda_c = lambda_c
        self.checkpoints_path = Path(checkpoints_path)
        self.pad_id = pad_id
        self.resume = resume

        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)
        self.metrics_tracker = TrackMetrics()
        self.nlg_metrics = Metrics()
        self.best_metric = 0
        self.current_epoch = 0

        self._setup_logging()

    def _setup_logging(self):
        """ Set up logging directories and TensorBoard writers. """
        time_tag = datetime.now().strftime("%d%m.%H%M") if not self.resume else Path(self.resume).stem
        self.checkpoints_path = self.checkpoints_path / time_tag
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)

        self.logger = SummaryWriter(log_dir=str(self.checkpoints_path / "logs"))
        self.loss_logger = SummaryWriter(log_dir=str(self.checkpoints_path / "loss"))
        self.bleu4_logger = SummaryWriter(log_dir=str(self.checkpoints_path / "bleu4"))
        self.gleu_logger = SummaryWriter(log_dir=str(self.checkpoints_path / "gleu"))

    def train_epoch(self, model: ModelType, data_loader: DataIterType):
        """ Run one training epoch. """
        model.train()
        for images, captions, lengths in tqdm(data_loader, desc="Training"):
            images, captions = images.to(self.device), captions.to(self.device)
            outputs = model(images, captions[:, :-1])  # ignore <EOS> token
            loss = self.criterion(outputs, captions[:, 1:])  # shift for <SOS>
            self.optims[0].zero_grad()  # Zero gradients for the optimizer
            loss.backward()  # Backpropagate the loss
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)  # Clip gradients
            self.optims[0].step()  # Update parameters

            # Log metrics
            self.metrics_tracker.update_running({'loss': loss.item()}, 'train')

        # Update global metrics after epoch
        self.metrics_tracker.update('train')

    def validate(self, model: ModelType, data_loader: DataIterType):
        """ Validate the model performance on the validation set. """
        model.eval()
        with torch.no_grad():
            for images, captions, lengths in tqdm(data_loader, desc="Validating"):
                images, captions = images.to(self.device), captions.to(self.device)
                outputs = model(images, captions[:, :-1])
                loss = self.criterion(outputs, captions[:, 1:])
                self.metrics_tracker.update_running({'loss': loss.item()}, 'val')

        self.metrics_tracker.update('val')

    def train(self, model: ModelType, train_loader: DataIterType, val_loader: DataIterType):
        """ Run the training and validation process. """
        for _ in range(self.epochs):
            self.train_epoch(model, train_loader)
            if self.current_epoch % self.val_interval == 0:
                self.validate(model, val_loader)
                # Implement early stopping, learning rate scheduler logic, and checkpoint saving here
            self.current_epoch += 1

if __name__ == "__main__":
    args = parse_arguments()
    seed_everything(42)  # Ensuring reproducibility

    # Model and optimizer setup
    model = nn.Sequential()  # Example: Define your model here
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Data loading
    train_loader = None  # Define your data loader
    val_loader = None  # Define your validation data loader

    # Initialize trainer
    trainer = Trainer([optimizer], [scheduler], torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                      epochs=10, val_interval=1, early_stop=5, lr_patience=3, embeddings_finetune=5,
                      grad_clip=1.0, lambda_c=0.1, checkpoints_path="/path/to/save", pad_id=0)
    trainer.train(model, train_loader, val_loader)
