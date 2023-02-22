# This module allows to run a example base experiment. All the experiments are in the notebooks folder.
# This module needs the package and an enviromnent to be installed. To install the package, run the following command:
#       $ pip install -e .
# Can't be run without the precomputed embeddings, but those are huge (>1GB), so I couldn't send them to you or put them on github.

import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks, seed_everything
from torch import nn

from nlp_assemblee.simple_datasets import AssembleeDataset
from nlp_assemblee.simple_trainer import LitModel, load_embedding, process_predictions
from nlp_assemblee.simple_visualisation import (
    calculate_metrics_binary,
    plot_confusion_matrix,
    plot_network_graph,
    plot_precision_recall_curve_binary,
    plot_roc_curve_binary,
)

if __name__ == "__main__":
    seed_everything(42, workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Definition of the parameters
    print("Loading parameters ...")
    FEATURES = True
    TEXT_VARS = ["intervention", "titre_regexed", "contexte"]
    DROP_CENTER = True

    BATCH_SIZE = 128
    MAX_EPOCHS = 100

    CALLBACKS = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0.005,
            check_finite=True,
            patience=15,
            verbose=True,
        ),
        callbacks.ModelSummary(max_depth=-1),
        callbacks.Timer(duration="00:03:00:00", interval="epoch"),
        callbacks.RichProgressBar(
            theme=callbacks.progress.rich_progress.RichProgressBarTheme(
                description="green_yellow", progress_bar="green1"
            )
        ),
        callbacks.LearningRateMonitor(logging_interval="epoch", log_momentum=False),
        callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
            save_last=True,
        ),
    ]

    OPTIMIZER_TYPE = "AdamW"
    OPTIMIZER_KWARGS = {}
    LR = 1e-4
    LOSS = "CrossEntropyLoss"

    SCHEDULER_KWARGS = {
        "scheduler": "ExponentialLR",
        "gamma": 0.95,
    }

    # Doesn't change between experiments
    LABEL_VAR = "label"
    DATA_ROOT = "./data/"
    NUM_WORKERS = 12
    PREFETCH_FACTOR = 4
    PIN_MEMORY = True
    ACCELERATOR = "gpu"
    DEVICE = "cuda"
    LOG_EVERY_N_STEPS = 50
    CHECK_VAL_EVERY_N_EPOCH = 1
    DETERMINISTIC = False

    MODEL_NAME = "camembert-base"
    MODEL_FOLDER = f"./data/precomputed/{MODEL_NAME}"
    RESULTS_PATH = f"./results/{MODEL_NAME}_ex/"
    LOGGER = pl.loggers.TensorBoardLogger(RESULTS_PATH, log_graph=False)
    Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)

    print("Batch size: ", BATCH_SIZE)
    print("Max epochs: ", MAX_EPOCHS)
    print("Optimizer: ", OPTIMIZER_TYPE)
    print("Scheduler: ", SCHEDULER_KWARGS["scheduler"])
    print("Results path: ", RESULTS_PATH)

    # Definition of the network
    print("Loading net ...")

    class Net(nn.Module):
        def __init__(self, root, embed_dim, inter_dim, dropout=0.2, freeze=True):
            super().__init__()
            self.example_input_array = {
                "text": {
                    "intervention": torch.randn(32, 768),
                    "titre_regexed": torch.randint(100, (32,)).int(),
                    "contexte": torch.randint(100, (32,)).int(),
                }
            }

            self.embed_dim = embed_dim
            self.inter_dim = inter_dim
            self.dropout = dropout
            self.freeze = freeze

            self.titre_embeddings = load_embedding(root, "titre_regexed", freeze=freeze)
            self.titre_fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed_dim, inter_dim),
                nn.ReLU(),
            )

            self.contexte_embeddings = load_embedding(root, "contexte", freeze=freeze)
            self.contexte_fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed_dim, inter_dim),
                nn.ReLU(),
            )

            self.intervention_fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed_dim, inter_dim),
                nn.ReLU(),
            )

            self.mlp = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(inter_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 2),
            )

        def forward(self, **x):
            intervention = x["text"]["intervention"]
            titre_regexed = x["text"]["titre_regexed"]
            contexte = x["text"]["contexte"]

            intervention_repr = self.intervention_fc(intervention)

            titre_emb = self.titre_embeddings(titre_regexed)
            titre_repr = self.titre_fc(titre_emb)

            contexte_emb = self.contexte_embeddings(contexte)
            contexte_repr = self.contexte_fc(contexte_emb)

            pooled_repr = intervention_repr + titre_repr + contexte_repr

            logits = self.mlp(pooled_repr)

            return logits

    NET = Net(MODEL_FOLDER, 768, 1024, dropout=0.2, freeze=True)

    # Training
    print("Training")
    lit_model = LitModel(
        NET,
        optimizer_type=OPTIMIZER_TYPE,
        learning_rate=LR,
        optimizer_kwargs=OPTIMIZER_KWARGS,
        scheduler_kwargs=SCHEDULER_KWARGS,
        criterion_type=LOSS,
        batch_size=BATCH_SIZE,
        loader_kwargs={
            "root": MODEL_FOLDER,
            "text_vars": TEXT_VARS,
            "use_features": FEATURES,
            "drop_center": DROP_CENTER,
            "label_var": LABEL_VAR,
            "num_workers": NUM_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
            "pin_memory": PIN_MEMORY,
        },
    )

    trainer = pl.Trainer(
        accelerator=ACCELERATOR,
        max_epochs=MAX_EPOCHS,
        logger=LOGGER,
        callbacks=CALLBACKS,
        deterministic=DETERMINISTIC,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
    )

    trainer.fit(lit_model)

    # Evaluation
    print("Evaluation on test set...")
    #### Prediction on test set
    preds = trainer.predict(ckpt_path="best")

    #### Metrics and logs
    results = process_predictions(preds)

    metrics = calculate_metrics_binary(results)
    print("Metrics: ", metrics)

    print(f"Saving logs in {RESULTS_PATH}")
    with open(Path(RESULTS_PATH) / "metrics.json", "w") as f:
        json.dump(metrics, f)

    logs_dict = {
        "last_epoch": trainer.current_epoch,
        "log_dir": trainer.log_dir,
        "ckpt_path": trainer.ckpt_path,
        "total_parameters": pl.utilities.model_summary.summarize(lit_model).total_parameters,
        "trainable_parameters": pl.utilities.model_summary.summarize(
            lit_model
        ).trainable_parameters,
        "model_size": pl.utilities.model_summary.summarize(lit_model).model_size,
        "hparams": dict(lit_model.hparams_initial),
        "time_elapsed": trainer.callbacks[2].time_elapsed(),
        "metrics": metrics,
    }

    with open(Path(RESULTS_PATH) / "logs.json", "w") as f:
        json.dump(logs_dict, f)

    #### Plots
    print("Plotting...")
    print(f"Saving plots in {RESULTS_PATH}")
    confusion_fig = plot_confusion_matrix(results, figsize=(6, 6), normalized=None)
    confusion_fig.savefig(Path(RESULTS_PATH) / "confusion_matrix.png")

    confusion_true_fig = plot_confusion_matrix(results, figsize=(6, 6), normalized="true")
    confusion_true_fig.savefig(Path(RESULTS_PATH) / "confusion_matrix_true_normed.png")

    roc_fig = plot_roc_curve_binary(results, figsize=(6, 6), palette="deep")
    roc_fig.savefig(Path(RESULTS_PATH) / "roc_curve.png")

    pr_fig = plot_precision_recall_curve_binary(results, figsize=(6, 6), palette="deep")
    pr_fig.savefig(Path(RESULTS_PATH) / "precision_recall_curve.png")

    network_fig = plot_network_graph(NET, device=DEVICE, model_name=MODEL_NAME, path=RESULTS_PATH)
