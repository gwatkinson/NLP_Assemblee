import json
from typing import Dict, List

import torch
from torch import nn
from transformers import BertModel, CamembertModel


class BertLinear(nn.Module):
    """Linear layer after a BERT layer."""

    def __init__(self, bert_type: str, frozen: bool, linear_dim: int, name: str = None) -> None:
        super().__init__()
        self.bert_type = bert_type
        self.frozen = frozen
        self.linear_dim = linear_dim
        self.name = name or f"{'frozen_' if frozen else ''}{bert_type}_linear_{linear_dim}"
        if bert_type == "bert":
            self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        elif bert_type == "camembert":
            self.bert = CamembertModel.from_pretrained("camembert-base")
        if frozen:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.bert_dim = self.bert.config.hidden_size
        if linear_dim > 0:
            self.linear = nn.Linear(self.bert_dim, linear_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pooled_output = self.bert(
            input_ids=input_ids, attention_mask=(input_ids != 0).long(), return_dict=True
        )[
            "pooler_output"
        ]  # (batch_size, nb_int, bert_dim)

        if self.linear_dim > 0:
            pooled_output = self.linear(pooled_output)

        return pooled_output


class BertLinears(nn.Module):
    """Creates multiple BertLinear layers."""

    def __init__(self, name: str = None, **bert_layers: Dict[str, BertLinear]) -> None:
        super().__init__()
        self.bert_layers = nn.ModuleDict(bert_layers)
        self.name = name or f"bert_linears_{list(bert_layers.keys())}"

    def forward(self, **inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        return [self.bert_layers[name](inputs[name]) for name in self.bert_layers]


class LinearPooler(nn.Module):
    """Concatenate the outputs of the linear layers."""

    def __init__(self, concat_type: str, name: str = None) -> None:
        super().__init__()
        self.concat_type = concat_type
        self.name = name or f"pooler_{concat_type}"

    def forward(self, *linear_outputs: List[torch.Tensor]) -> torch.Tensor:
        if self.concat_type == "mean":
            return torch.mean(torch.stack(linear_outputs), dim=0)
        elif self.concat_type == "concat":
            return torch.cat(linear_outputs, dim=1)


class MLPLayer(nn.Module):
    """Multi-layer perceptron classifier."""

    def __init__(
        self,
        mlp_dims: List[int],
        dropout: float = 0.1,
        negative_slope: float = 0.01,
        batch_norm: bool = True,
        name: str = None,
    ) -> None:
        super().__init__()
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.name = name or f"mlp_{mlp_dims}_{dropout}"

        self.layers = nn.ModuleList()
        for i in range(len(mlp_dims) - 1):
            if dropout > 0 and dropout < 1:
                self.layers.append(nn.Dropout(dropout))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(mlp_dims[i]))
            self.layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            self.layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Classifier(nn.Module):
    """Class that combines the BertLinears, the pooler and the MLP."""

    def __init__(
        self,
        bert_layers: Dict[str, BertLinear],
        features_mlp: MLPLayer | bool,
        pooler: LinearPooler,
        mlp: MLPLayer,
        name: str = None,
    ) -> None:
        super().__init__()
        self.inputs_keys = list(bert_layers.keys())
        self.bert_linears = BertLinears(**bert_layers)
        self.features_mlp = features_mlp
        self.pooler = pooler
        self.mlp = mlp
        self.name = name or f"classifier_{self.bert_linears.name}_{self.mlp.name}"

    def forward(self, **inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        keys = list(inputs.keys())

        bert_inputs = {key: inputs[key] for key in keys if key != "features"}

        bert_outputs = self.bert_linears(**bert_inputs)

        if self.features_mlp:
            features_input = inputs["features"]
            features_output = self.features_mlp(features_input)
            pooled_output = self.pooler(*bert_outputs, features_output)
        else:
            pooled_output = self.pooler(*bert_outputs)

        pred = self.mlp(pooled_output)

        return pred


def build_classifier_from_config(conf_file):
    with open(conf_file, "r") as f:
        conf = json.load(f)["classifier"]

    bert_linears = {
        name: BertLinear(**conf["linear_layers"]["layers"][name])
        for name in conf["linear_layers"]["layers"]
    }
    if conf["feature_mlp"]:
        features_mlp = MLPLayer(**conf["feature_mlp"])
    else:
        features_mlp = False
    pooler = LinearPooler(**conf["pooler_layer"])
    mlp = MLPLayer(**conf["mlp_layer"])

    classifier = Classifier(bert_linears, features_mlp, pooler, mlp, conf["name"])

    return classifier
