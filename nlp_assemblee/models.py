# Copyright (c) 2022 Gabriel WATKINSON and Jéremie STYM-POPPER
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel, CamembertModel


class BertLinear(nn.Module):
    """Linear layer after a BERT layer."""

    def __init__(self, bert_type, frozen, linear_dim, name=None):
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

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )  # (batch_size, nb_int, bert_dim)

        if self.linear_dim > 0:
            pooled_output = self.linear(pooled_output)

        return pooled_output


class ConcatLayer(nn.Module):
    """Concatenate the outputs of the BertLinear layers."""

    def __init__(self, concat_type, name=None):
        self.concat_type = concat_type
        self.name = name or f"concat_{concat_type}"

    def forward(self, *bert_linear_outputs):
        if self.concat_type == "mean":
            return torch.mean(torch.stack(bert_linear_outputs), dim=0)
        elif self.concat_type == "concat":
            return torch.cat(bert_linear_outputs, dim=2)


class SimpleCamembertClassifier(nn.Module):
    def __init__(
        self,
        bert_model,
        num_classes=11,
        bert_dim=768,
        input_dim=256,
        embed_dim=256,
        input_dim2=256,
        num_heads=8,
        dropout=0.2,
    ):
        super(SimpleCamembertClassifier, self).__init__()

        # Set the parameters
        self.bert_dim = bert_dim
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.input_dim2 = input_dim2
        self.num_attention_heads = num_heads
        self.attention_head_dim = input_dim // num_heads

        # Initialize the BERT model
        self.bert_model = bert_model

        # Initialize the linear layers for the multi-head attention
        self.query_linear = nn.Linear(bert_dim, input_dim)
        self.key_linear = nn.Linear(bert_dim, input_dim)
        self.value_linear = nn.Linear(bert_dim, input_dim)

        # Initialize the multi-head attention layer
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True
        )

        # Initialize the linear layers for the output projection
        self.linear1 = nn.Linear(input_dim, input_dim2)
        self.linear2 = nn.Linear(input_dim, num_classes)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, tokens, masks, interventions_masks):
        """tokens (batch_size, nb_int, nb_tokens) masks (batch_size, nb_int,
        nb_tokens) interventions_masks (batch_size, nb_int)"""
        pooled_output = self.bert_model(
            input_ids=tokens, attention_mask=masks, return_dict=False
        )  # (batch_size, nb_int, bert_dim)

        # Add a zero padding for CLS in attention
        pooled_output_with_cls = F.pad(
            pooled_output, pad=(0, 0, 1, 0), mode="constant", value=0.0
        )  # (batch_size, nb_int + 1, bert_dim)

        # Split the input tensors into the different attention heads
        query_heads = self.query_linear(
            pooled_output_with_cls
        )  # (batch_size, nb_int + 1, input_dim)
        key_heads = self.key_linear(pooled_output_with_cls)  # (batch_size, nb_int + 1, input_dim)
        value_heads = self.value_linear(pooled_output_with_cls)  # (batch_size, input_dim)

        # Apply the multi-head attention
        attn_output = self.attention_layer(
            query_heads,
            key_heads,
            value_heads,
            key_padding_mask=interventions_masks,
            need_weights=False,
        )  # (batch_size, input_dim)

        deputy_repr = attn_output[:, 0, :]  # (batch_size, input_dim)

        dropout_output1 = self.dropout1(deputy_repr)
        linear_output1 = nn.ReLU(self.linear1(dropout_output1))  # (batch_size, input_dim2)
        dropout_output2 = self.dropout2(linear_output1)
        linear_output2 = nn.ReLU(self.linear2(dropout_output2))  # (batch_size, num_classes)

        final_layer = nn.Softmax(linear_output2)  # (batch_size, num_classes)

        return final_layer
