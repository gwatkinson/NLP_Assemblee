import pickle
from pathlib import Path

from torch.utils.data import DataLoader, Dataset


class AssembleeDataset(Dataset):
    def __init__(self, root, bert_type, phase, text_vars, use_features, label_var):
        super().__init__()
        self.path = Path(root) / f"{bert_type}_embeddings_{phase}.pkl"
        self.records = self.load_records()

        self.labels = self.records[label_var]

        self.use_features = use_features
        if use_features:
            self.features = self.records["features"]
        else:
            self.features = False

        self.text_vars = text_vars
        if text_vars:
            self.text = {var: self.records[var] for var in text_vars}
        else:
            self.text = False

    def load_records(self):
        with open(self.path, "rb") as f:
            records = pickle.load(f)
        return records

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return self.labels[idx]

    def get_batch_inputs(self, idx):
        inputs = {}
        if self.text_vars:
            inputs["text"] = {var: self.text[var][idx] for var in self.text_vars}
        if self.use_features:
            inputs["features"] = self.features[idx].astype(float)
        return inputs

    def __getitem__(self, idx):
        batch_x = self.get_batch_inputs(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y


def get_single_dataloader(
    root,
    phase,
    bert_type,
    batch_size,
    text_vars,
    use_features,
    label_var="label",
    num_workers=2,
    prefetch_factor=3,
    pin_memory=True,
):
    dataset = AssembleeDataset(
        root=root,
        bert_type=bert_type,
        phase=phase,
        text_vars=text_vars,
        use_features=use_features,
        label_var=label_var,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    return dataset, loader, len(dataset)


def get_dataloader(
    root,
    bert_type,
    batch_size,
    text_vars,
    use_features,
    label_var="label",
    num_workers=2,
    prefetch_factor=3,
    pin_memory=True,
):
    trainset = AssembleeDataset(
        root=root,
        bert_type=bert_type,
        phase="train",
        text_vars=text_vars,
        use_features=use_features,
        label_var=label_var,
    )
    testset = AssembleeDataset(
        root=root,
        bert_type=bert_type,
        phase="test",
        text_vars=text_vars,
        use_features=use_features,
        label_var=label_var,
    )
    valset = AssembleeDataset(
        root=root,
        bert_type=bert_type,
        phase="val",
        text_vars=text_vars,
        use_features=use_features,
        label_var=label_var,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    datasets = {"train": trainset, "test": testset, "val": valset}

    lengths = {"train": len(trainset), "test": len(testset), "val": len(valset)}

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return datasets, loaders, lengths
