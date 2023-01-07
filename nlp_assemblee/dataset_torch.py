# Copyright (c) 2022 Gabriel WATKINSON and Jéremie STYM-POPPER
# SPDX-License-Identifier: MIT

# model Bert

import numpy as np
import torch
from transformers import CamembertTokenizer

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
tokenizer.pad_token = "[PAD]"

labels = {
    "LR": 0,
    "GDR": 1,
    "REN": 2,
    "RN": 3,
    "MODEM": 4,
    "LFI": 5,
    "SOC": 6,
    "ECO": 7,
    "HOR": 8,
    "LIOT": 9,
    "NI": 10,
}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, labels=labels, tokenizer=tokenizer, max_length=200):

        self.labels = [labels[label] for label in df["groupe"]]
        self.texts = [
            tokenizer(
                interventions,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            for interventions in df["interventions"]
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
