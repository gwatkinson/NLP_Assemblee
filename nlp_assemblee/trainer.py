import json

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn

from nlp_assemblee.datasets import build_dataset_and_dataloader_from_config
from nlp_assemblee.models import build_classifier_from_config


class LitModel(pl.LightningModule):
    def __init__(self, training_parameters):
        super().__init__()
        self.classifier = training_parameters["model"]
        self.criterion = training_parameters["criterion"]
        self.training_parameters = training_parameters

    def forward(self, x):
        return self.classifier(**x)

    def configure_optimizers(self):
        optimizer = self.training_parameters["optimizer"]
        # scheduler = self.training_parameters["scheduler"]
        return optimizer

    def get_loss(self, batch, model_type="train"):
        x, y = batch
        z = self.classifier(**x)
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
        "model": model,
        "seed": seed,
        "optimizer": optimizer,
        "criterion": criterion,
        "epochs": num_epochs,
        "precision": precision,
        "scheduler": scheduler,
        "list_metrics": list_metrics,
        "tensorboard_dir": tensorboard,
        "checkpoint": [checkpoint],
        "earlystop": [earlystop],
    }

    lit_model = LitModel(training_parameters)

    return lit_model, training_parameters


def perform_lightning(path_conf_file):
    datasets, loaders = build_dataset_and_dataloader_from_config(path_conf_file)
    lit_model, training_parameters = build_trainer_from_config(path_conf_file)

    if training_parameters["seed"]:
        torch.manual_seed(training_parameters["seed"])

    # Tensorboard config
    tensorboard = training_parameters["tensorboard"]

    # Checkpoint config
    checkpoint = training_parameters["checkpoint"]

    # EarlyStopping config
    earlystop = training_parameters["earlystop"]

    callbacks = checkpoint + earlystop
    trainer = pl.Trainer(logger=tensorboard, callbacks=callbacks)
    trainer.fit(lit_model, loaders["train"], loaders["val"])

    return trainer
