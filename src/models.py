from torch import nn
from transformers import CamembertModel

bert = CamembertModel.from_pretrained('camembert-base')

class CamembertClassifier(nn.Module):

    def __init__(self, bert=bert, dropout=0.5, num_classes=11):

        super(BertClassifier, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        mean_interventions = pooled_output.mean(dim=-2)
        dropout_output = self.dropout(mean_interventions)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
