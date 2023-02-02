import json

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn

from models import build_classifier_from_config


class LitModel(pl.LightningModule):
    def __init__(
        self,
        # classifier: nn.Module,
        path_conf_file: str,
    ):
        super().__init__()
        self.model = build_classifier_from_config(path_conf_file)

        self.train_parameters = build_trainer_from_config(path_conf_file)
        self.criterion = self.train_parameters["loss"]

    def forward(self, x):
        return self.model(**x)

    def configure_optimizers(self):
        optimizer = self.train_parameters["optimizer"]
        lr_scheduler = self.train_parameters["scheduler"]
        return [optimizer], [lr_scheduler]

    def get_loss(self, batch, model_type="train"):
        x, y = batch
        z = self.model(**x)
        loss = self.criterion(z, y)
        self.log(f"{model_type}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        tain_loss = self.get_loss(batch, model_type="train")
        return tain_loss

    def validation_step(self, val_batch, batch_idx):
        val_loss = self.get_loss(val_batch, model_type="val")
        return val_loss

    def testing_step(self, val_batch, batch_idx):
        test_loss = self.get_loss(val_batch, model_type="test")
        return test_loss


def build_trainer_from_config(conf_file):
    model = build_classifier_from_config(conf_file)

    with open(conf_file, "r") as f:
        conf = json.load(f)["trainer"]

    # Global config
    num_epochs = conf["epochs"]
    precision = conf["precision"]
    list_metrics = conf["metrics"]
    seed = conf["seed"]

    # Optimizer config
    if conf["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **conf["optimizer_kwargs"])
    elif conf["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), **conf["optimizer_kwargs"])

    # Loss config
    if conf["loss"] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(**conf["loss_kwargs"])
    elif conf["loss"] == "MSEloss":
        criterion = nn.MSELoss(**conf["loss_kwargs"])

    # Scheduler confg
    if conf["scheduler"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **conf["scheduler_kwargs"]
        )

    # Tensorboard config
    if conf["tensorboard"]:
        tensorboard = pl_loggers.TensorBoardLogger(**conf["tensorboard_kwargs"])
    else:
        tensorboard = False

    # Checkpoint config
    if conf["checkpoint"]:
        checkpoint = ModelCheckpoint(**conf["checkpoint_kwargs"])
    else:
        checkpoint = None

    # EarlyStopping config
    if conf["early_stopping"]:
        earlystop = EarlyStopping(**conf["early_stopping_kwargs"])
    else:
        earlystop = None

    training_parameters = {
        "seed": seed,
        "optimizer": optimizer,
        "loss": criterion,
        "epochs": num_epochs,
        "precision": precision,
        "scheduler": scheduler,
        "list_metrics": list_metrics,
        "tensorboard_dir": tensorboard,
        "checkpoint": [checkpoint],
        "earlystop": [earlystop],
    }

    return training_parameters


def perform_lightning(lightning_model, train_loader, val_loader, path_conf_file):
    model = lightning_model()
    trainer_parameters = build_trainer_from_config(path_conf_file)

    if trainer_parameters["seed"]:
        torch.manual_seed(trainer_parameters["seed"])

    # Tensorboard config
    tensorboard = trainer_parameters["tensorboard"]

    # Checkpoint config
    checkpoint = trainer_parameters["checkpoint"]

    # EarlyStopping config
    earlystop = trainer_parameters["earlystop"]

    callbacks = checkpoint + earlystop
    trainer = pl.Trainer(logger=tensorboard, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)

    return None
