# Copyright (c) 2022 Gabriel WATKINSON and Jéremie STYM-POPPER
# SPDX-License-Identifier: MIT

from torch import nn
from transformers import CamembertModel

# from transformers import BertModel

# bert = BertModel.from_pretrained("bert-base-multilingual-cased")
camembert_model = CamembertModel.from_pretrained("camembert-base")


class CamembertClassifier(nn.Module):
    def __init__(
        self,
        num_classes=11,
        bert_model=camembert_model,
        bert_dim=768,
        input_dim=768,
        num_heads=5,
        dropout=0.2,
    ):
        super(CamembertClassifier, self).__init__()

        # Set the parameters
        self.bert_dim = bert_dim
        self.input_dim = input_dim
        self.num_attention_heads = num_heads
        self.attention_head_dim = input_dim // num_heads

        # Initialize the BERT model
        self.bert_model = bert_model

        # Initialize the linear layers for the multi-head attention
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)

        # Initialize the multi-head attention layer
        self.attention_layer = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

        # Initialize the linear layers for the output projection
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        pooled_output = self.bert_model(input_ids=input_id, attention_mask=mask, return_dict=False)

        # Split the input tensors into the different attention heads
        query_heads = self.query_linear(pooled_output)
        key_heads = self.key_linear(pooled_output)
        value_heads = self.value_linear(pooled_output)

        # Apply the multi-head attention
        attn_output, attn_output_weights = self.attention_layer(
            query_heads, key_heads, value_heads, mask=mask
        )

        dropout_output = self.dropout(attn_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
