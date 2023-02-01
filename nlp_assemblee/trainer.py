import pytorch_lightning as pl
import numpy as np
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from transformers import CamembertModel, CamembertTokenizer, BertModel, BertTokenizer
from models import build_classifier_from_config, build_trainer_from_config
from pytorch_lightning import loggers as pl_loggers


class LitModel(pl.LightningModule):

    def __init__(
        self,
        # classifier: nn.Module,
        path_conf_file : str,
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
        loss = self.get_loss(batch, model_type="train")
        return loss

    def validation_step(self, val_batch, batch_idx):
        self.get_loss(val_batch, model_type="val")
        return None 

def perform_lightning(lightning_model, train_loader, val_loader, path_conf_file):

    model = lightning_model()
    trainer_parameters = build_trainer_from_config(path_conf_file)
    gpus = int(torch.cuda.is_available())

    # Tensorboard config
    tensorboard = trainer_parameters["tensorboard"]

    trainer = pl.Trainer(logger=tensorboard, gpus=gpus)

    pass 

class SeanceLitClassifier(pl.LightningModule):
    def __init__(
        self,
        processed_df,
        labels_dict,
        batch_size=32,
        num_workers=8,
        type_bert="camembert",
        freeze_bert=True,
        intervention_dim=256,
        titre_dim=128,
        profession_dim=64,
        bert_dim=768,
        dropout=0.1,
        lr=1e-4,
    ):
        super().__init__()
        self.example_input_array = {
            "intervention": torch.randint(0, 100, (32, 1, 512)),
            "profession": torch.randint(0, 100, (32, 1, 16)),
        }

        # Parameters
        self.num_classes = len(np.unique(list(labels_dict.values())))
        self.processed_df = processed_df
        self.labels_dict = labels_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.type_bert = type_bert
        self.freeze_bert = freeze_bert
        self.intervention_dim = intervention_dim
        self.titre_dim = titre_dim
        self.profession_dim = profession_dim
        self.bert_dim = bert_dim
        self.dropout = dropout
        self.lr = lr

        # Models
        if type_bert == "camembert":
            self.bert_model = CamembertModel.from_pretrained("camembert-base")
            self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        else:
            self.bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

        if freeze_bert:
            for p in self.bert_model.parameters():
                p.requires_grad = False

        self.classifier = SeanceClassifier(
            bert_model=self.bert_model,
            num_classes=self.num_classes,
            intervention_dim=intervention_dim,
            titre_dim=titre_dim,
            profession_dim=profession_dim,
            bert_dim=bert_dim,
            dropout=dropout,
        )

        # Save the hyperparameters
        # See https://pytorch-lightning.readthedocs.io/en/latest/common/checkpointing_basic.html#save-a-checkpoint
        self.save_hyperparameters()

    def forward(self, intervention, titre, profession):
        return self.classifier(intervention, titre, profession)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x_hat = self.forward(x["intervention"], x["titre"], x["profession"])
        loss = nn.CrossEntropyLoss()(x_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        # See more https://pytorch-lightning.readthedocs.io/en/latest/common/evaluation_basic.html
        x, y = batch
        x_hat = self.forward(x["intervention"], x["titre"], x["profession"])
        test_loss = nn.CrossEntropyLoss()(x_hat, y)
        # Logging to TensorBoard by default
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x_hat = self.forward(x["intervention"], x["titre"], x["profession"])
        val_loss = nn.CrossEntropyLoss()(x_hat, y)
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def prepare_data(self):
        self.train_data, self.val_data, self.test_data = get_dataset(
            self.tokenizer,
            self.processed_df,
            self.labels_dict,
            group_var="groupe",
            intervention_var="intervention",
            titre_var="titre_complet",
            profession_var="profession",
            max_len_padding=512,
            max_len_padding_titre=64,
            max_len_padding_profession=16,
            test_frac=0.25,
            val_frac=0.2,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size
        )

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.batch_size)
