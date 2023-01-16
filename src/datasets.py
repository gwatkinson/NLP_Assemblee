import numpy as np
import torch
from torch.utils.data import Dataset


class SeanceDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        df,
        labels_dict,
        group_var="groupe",
        intervention_var="intervention",
        titre_var="titre_complet",
        profession_var="profession",
        max_len_padding=512,
        max_len_padding_titre=64,
        max_len_padding_profession=16,
    ):
        # Parameters
        self.df = df
        self.labels_dict = labels_dict
        self.inverse_label_dict = {v: k for k, v in labels_dict.items()}
        self.group_var = group_var
        self.intervention_var = intervention_var
        self.titre_var = titre_var
        self.profession_var = profession_var
        self.max_len_padding = max_len_padding
        self.max_len_padding_titre = max_len_padding_titre
        self.max_len_padding_profession = max_len_padding_profession

        # Inputs and labels
        self.labels = [labels_dict[label] for label in df[group_var]]

        self.interventions = [
            tokenizer(
                text,
                padding="max_length",
                max_length=max_len_padding,
                truncation=True,
                return_tensors="pt",
            )
            for text in df[intervention_var]
        ]

        self.titres = [
            tokenizer(
                text,
                padding="max_length",
                max_length=max_len_padding_titre,
                truncation=True,
                return_tensors="pt",
            )
            for text in df[titre_var]
        ]

        self.professions = [
            tokenizer(
                text,
                padding="max_length",
                max_length=max_len_padding_profession,
                truncation=True,
                return_tensors="pt",
            )
            for text in df[profession_var]
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_inputs(self, idx):
        return {
            "intervention": self.interventions[idx],
            "titre": self.titres[idx],
            "profession": self.professions[idx],
        }

    def __getitem__(self, idx):
        batch_x = self.get_batch_inputs(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_x, batch_y


def get_dataset(
    tokenizer,
    processed_df,
    labels_dict,
    group_var="groupe",
    intervention_var="intervention",
    titre_var="titre_complet",
    profession_var="profession",
    max_len_padding=512,
    max_len_padding_titre=64,
    max_len_padding_profession=16,
    test_frac=0.25,
    val_frac=0.2,
):
    X, y = np.arange(len(processed_df)), processed_df["groupe"]
    test_frac = 0.25
    val_frac = 0.2

    idx_train, idx_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=42, stratify=y
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_train, y_train, test_size=val_frac, random_state=42, stratify=y_train
    )

    train_dataset = SeanceDataset(
        tokenizer,
        processed_df.iloc[idx_train],
        labels_dict,
        group_var=group_var,
        intervention_var=intervention_var,
        titre_var=titre_var,
        profession_var=profession_var,
        max_len_padding=max_len_padding,
        max_len_padding_titre=max_len_padding_titre,
        max_len_padding_profession=max_len_padding_profession,
    )

    val_dataset = SeanceDataset(
        tokenizer,
        processed_df.iloc[idx_val],
        labels_dict,
        group_var=group_var,
        intervention_var=intervention_var,
        titre_var=titre_var,
        profession_var=profession_var,
        max_len_padding=max_len_padding,
        max_len_padding_titre=max_len_padding_titre,
        max_len_padding_profession=max_len_padding_profession,
    )

    test_dataset = SeanceDataset(
        tokenizer,
        processed_df.iloc[idx_test],
        labels_dict,
        group_var=group_var,
        intervention_var=intervention_var,
        titre_var=titre_var,
        profession_var=profession_var,
        max_len_padding=max_len_padding,
        max_len_padding_titre=max_len_padding_titre,
        max_len_padding_profession=max_len_padding_profession,
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloader(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=32,
    num_workers=8,
):
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader
