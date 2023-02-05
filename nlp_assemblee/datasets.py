import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class AssembleeDataset(Dataset):
    def __init__(self, records, bert_type, text_vars, features_vars, label_var):
        self.records = records
        self.bert_type = bert_type
        self.text_vars = text_vars
        self.features_vars = features_vars
        self.label_var = label_var

        self.max_len_padding = 512

    def __len__(self):
        return len(self.records)

    def get_batch_labels(self, idx):
        return self.records[idx][self.label_var]

    def get_batch_inputs(self, idx):
        inputs = {var: self.records[idx][f"{self.bert_type}_tokens"][var] for var in self.text_vars}
        if self.features_vars:
            float_inputs = [float(self.records[idx][var]) for var in self.features_vars]
            inputs["features"] = np.array(float_inputs)

        return inputs

    def __getitem__(self, idx):
        batch_x = self.get_batch_inputs(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_x, batch_y


def collate_fn(data):
    """
    data: is a list of tuples with (example, label, length)
          where 'example' is a tensor of arbitrary shape
          and label/length are scalars
    """
    labels = torch.tensor([int(x[1]) for x in data])

    padded_inputs = {}

    keys = data[0][0].keys()

    for var in keys:
        if var == "features":
            padded_inputs["features"] = torch.tensor(np.array([x[0][var] for x in data]))
        else:
            input_ids = pad_sequence([torch.tensor(x[0][var]) for x in data], batch_first=True)
            padded_inputs[var] = input_ids

    return padded_inputs, labels.long()


def load_records(records_path):
    with open(records_path, "rb") as f:
        records = pickle.load(f)
    return records


def build_dataset_and_dataloader_from_config(conf_file, path_prefix="./"):
    with open(conf_file, "r") as f:
        conf = json.load(f)["dataset"]

    path = Path(path_prefix) / conf["records_path"]

    records = load_records(path)

    X = np.arange(len(records))
    y = [record["groupe"] for record in records]
    idx_train, idx_test, y_train, y_test = train_test_split(
        X, y, test_size=conf["test_pc"], random_state=conf["random_state"], stratify=y
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_train,
        y_train,
        test_size=conf["val_pc"],
        random_state=conf["random_state"],
        stratify=y_train,
    )

    train_records = [records[idx] for idx in idx_train]
    test_records = [records[idx] for idx in idx_test]
    val_records = [records[idx] for idx in idx_val]

    train_dataset = AssembleeDataset(
        records=train_records,
        bert_type=conf["bert_type"],
        text_vars=conf["text_vars"],
        features_vars=conf["feature_vars"],
        label_var=conf["label_var"],
    )
    test_dataset = AssembleeDataset(
        records=test_records,
        bert_type=conf["bert_type"],
        text_vars=conf["text_vars"],
        features_vars=conf["feature_vars"],
        label_var=conf["label_var"],
    )
    val_dataset = AssembleeDataset(
        records=val_records,
        bert_type=conf["bert_type"],
        text_vars=conf["text_vars"],
        features_vars=conf["feature_vars"],
        label_var=conf["label_var"],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=conf["batch_size"],
        collate_fn=collate_fn,
        num_workers=conf["num_workers"],
        prefetch_factor=conf["prefetch_factor"],
        shuffle=conf["shuffle"],
        pin_memory=conf["pin_memory"],
        drop_last=conf["drop_last"],
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=conf["batch_size"],
        collate_fn=collate_fn,
        num_workers=conf["num_workers"],
        prefetch_factor=conf["prefetch_factor"],
        pin_memory=conf["pin_memory"],
        drop_last=False,
        shuffle=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=conf["batch_size"],
        collate_fn=collate_fn,
        num_workers=conf["num_workers"],
        prefetch_factor=conf["prefetch_factor"],
        pin_memory=conf["pin_memory"],
        drop_last=False,
        shuffle=False,
    )

    datasets = {"train": train_dataset, "test": test_dataset, "val": val_dataset}

    loaders = {"train": train_dataloader, "test": test_dataloader, "val": val_dataloader}

    return datasets, loaders
