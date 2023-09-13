import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class HateTargetBERT(nn.Module):
    def __init__(self, checkpoint, num_labels, rule_dimension=None): 
        super(HateTargetBERT, self).__init__() 
        self.num_labels = num_labels
        self.rule_dimension = rule_dimension
        #Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True))
        self.dropout1 = nn.Dropout(0.1, inplace=False) 
        self.classifier1 = nn.Linear(768 + self.rule_dimension, 128) # load and initialize weights
        self.dropout2 = nn.Dropout(0.1, inplace=False) 
        self.classifier2 = nn.Linear(128 + self.rule_dimension, 2) # load and initialize weights
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, rules=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = self.dropout1(outputs[0]) #outputs[0]=last hidden state
        sequence_output = sequence_output[:, 0, :].view(-1, 768)
        output_with_rules = torch.cat((sequence_output, rules), dim=1)
        output = self.classifier1(output_with_rules) # calculate losses
        output = self.dropout2(output)
        output = torch.cat((output, rules), dim=1)
        logits = self.classifier2(output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
