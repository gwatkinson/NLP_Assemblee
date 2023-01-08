# Copyright (c) 2022 Gabriel WATKINSON and Jéremie STYM-POPPER
# SPDX-License-Identifier: MIT

# model Bert

from torch.utils.data import Dataset


class InterventionsDataset(Dataset):
    def __init__(self, dict_data):
        self.labels = dict_data["labels"]
        self.masks = dict_data["masks"]
        self.tokens = dict_data["tokens"]
        self.interventions_masks = dict_data["interventions_masks"]

    def __len__(self):
        return len(self.labels)

    def get_batch_texts(self, idx):
        return {
            "tokens": self.tokens[idx],
            "masks": self.masks[idx],
            "interventions_masks": self.interventions_masks[idx],
        }

    def get_batch_labels(self, idx):
        return self.labels[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
