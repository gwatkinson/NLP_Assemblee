import pickle
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim

from nlp_assemblee.simple_datasets import get_single_dataloader


def load_embedding(path, var, freeze=True):
    folder_path = Path(path)
    f = folder_path / f"{var}/embeddings.pkl"
    assert f.exists()
    with open(f, "rb") as f:
        embs = pickle.load(f)["embeddings"]

    layer = nn.Embedding.from_pretrained(torch.tensor(embs), freeze=freeze)
    layer.name = f"Embedding_{var}"

    return layer


class LitModel(pl.LightningModule):
    def __init__(
        self,
        classifier,
        optimizer_type="Adam",
        learning_rate=1e-3,
        optimizer_kwargs={},
        scheduler_kwargs=None,
        criterion_type="CrossEntropyLoss",
        batch_size=256,
        loader_kwargs={
            "root": "../../../data/",
            "bert_type": "camembert",
            "text_vars": ["intervention"],
            "use_features": False,
            "label_var": "label",
            "num_workers": 12,
            "prefetch_factor": 4,
            "pin_memory": True,
        },
    ):
        super().__init__()

        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.criterion_type = criterion_type
        self.batch_size = batch_size
        self.loader_kwargs = loader_kwargs
        self.save_hyperparameters(ignore=["classifier", "criterion"])

        if criterion_type == "CrossEntropyLoss":
            self.criterion = nn.functional.cross_entropy

        self.classifier = classifier

        try:
            self.example_input_array = classifier.example_input_array
        except Exception:
            print("Didn't find example input array")

    def forward(self, **x):
        return self.classifier(**x)

    def get_optimizer_from_string(self):
        if self.optimizer_type == "SGD":
            optimizer = optim.SGD(
                self.classifier.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
            )
        elif self.optimizer_type == "Adam":
            optimizer = optim.Adam(
                self.classifier.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
            )
        elif self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                self.classifier.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
            )
        elif self.optimizer_type == "RMSprop":
            optimizer = optim.RMSprop(
                self.classifier.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
            )
        elif self.optimizer_type == "Adagrad":
            optimizer = optim.Adagrad(
                self.classifier.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
            )
        return optimizer

    def get_scheduler_from_string(self, optimizer):
        scheduler_type = self.scheduler_kwargs.pop("scheduler", "StepLR")
        if scheduler_type == "StepLR":
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_kwargs.pop("interval", 30),
                gamma=self.scheduler_kwargs.pop("gamma", 0.1),
            )
        elif scheduler_type == "ReduceLROnPlateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.scheduler_kwargs.pop("mode", "min"),
                factor=self.scheduler_kwargs.pop("factor", 0.1),
                patience=self.scheduler_kwargs.pop("patience", 10),
            )
        elif scheduler_type == "CosineAnnealingLR":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_kwargs.pop("T_max", 1000),
                eta_min=self.scheduler_kwargs.pop("eta_min", 0),
            )
        elif scheduler_type == "CyclicLR":
            lr_scheduler = optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.scheduler_kwargs.pop("base_lr", 1e-6),
                max_lr=self.scheduler_kwargs.pop("max_lr", 5e-3),
                step_size_up=self.scheduler_kwargs.pop("step_size_up", 2000),
                mode=self.scheduler_kwargs.pop("mode", "triangular2"),
            )
        elif scheduler_type == "OneCycleLR":
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.scheduler_kwargs.pop("max_lr", 5e-3),
                pct_start=self.scheduler_kwargs.pop("pct_start", 0.3),
                epochs=self.scheduler_kwargs.pop("epochs", 30),
                steps_per_epoch=self.scheduler_kwargs.pop("steps_per_epoch", 100),
            )
        elif scheduler_type == "ExponentialLR":
            lr_scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.scheduler_kwargs.pop("gamma", 0.1),
            )
        return lr_scheduler

    def configure_optimizers(self):
        optimizer = self.get_optimizer_from_string()

        if self.scheduler_kwargs is None:
            return optimizer

        lr_scheduler = self.get_scheduler_from_string(optimizer)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": self.scheduler_kwargs.pop("interval", "epoch"),
            "frequency": self.scheduler_kwargs.pop("frequency", 1),
            "strict": self.scheduler_kwargs.pop("strict", True),
            **self.scheduler_kwargs,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def get_loss(self, batch, model_type="train"):
        x, y = batch
        z = self.classifier(**x)
        loss = self.criterion(z, y)
        self.log(f"{model_type}_loss", loss, prog_bar=(model_type == "val"))

        _, predicted = z.max(1)
        accuracy = predicted.eq(y).sum().item() / y.size(0)
        self.log(f"{model_type}_accuracy", accuracy, prog_bar=(model_type == "val"))

        return loss

    def training_step(self, batch, batch_idx):
        tain_loss = self.get_loss(batch, model_type="train")
        return tain_loss

    def validation_step(self, val_batch, batch_idx):
        val_loss = self.get_loss(val_batch, model_type="val")
        return val_loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        z = self.classifier(**x)
        loss = self.criterion(z, y)
        self.log("test_loss", loss)

        _, predicted = z.max(1)
        accuracy = predicted.eq(y).sum().item() / y.size(0)
        self.log("test_accuracy", accuracy)

        return {"loss": loss, "accuracy": accuracy, "y": y, "z": z, "predicted": predicted}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        z = self.classifier(**x)
        _, predicted = z.max(1)

        return {
            "predicted": predicted,
            "logits": z,
            "probs": nn.functional.softmax(z, dim=1),
            "labels": y,
        }

    def train_dataloader(self):
        _, loader, _ = get_single_dataloader(
            phase="train", batch_size=self.batch_size, **self.loader_kwargs
        )
        return loader

    def val_dataloader(self):
        _, loader, _ = get_single_dataloader(
            phase="val", batch_size=self.batch_size, **self.loader_kwargs
        )
        return loader

    def test_dataloader(self):
        _, loader, _ = get_single_dataloader(
            phase="test", batch_size=self.batch_size, **self.loader_kwargs
        )
        return loader

    def predict_dataloader(self):
        _, loader, _ = get_single_dataloader(
            phase="test", batch_size=self.batch_size, **self.loader_kwargs
        )
        return loader


def process_predictions(outputs):
    labels = []
    logits = []
    probs = []
    predictions = []
    for batch_output in outputs:
        labels.extend(batch_output["labels"].tolist())
        logits.append(batch_output["logits"].cpu().numpy())
        probs.append(batch_output["probs"].cpu().numpy())
        predictions.extend(batch_output["predicted"].tolist())
    labels = np.array(labels)
    predictions = np.array(predictions)
    logits = np.vstack(logits)
    probs = np.vstack(probs)
    return {"labels": labels, "logits": logits, "probs": probs, "predictions": predictions}
