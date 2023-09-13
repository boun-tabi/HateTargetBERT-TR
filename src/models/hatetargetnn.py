import torch.nn as nn
import torch.nn.functional as F


class HateTargetNN(nn.Module):
    def __init__(self, num_labels, rule_dimension=None): 
        super(HateTargetNN, self).__init__() 
        self.num_labels = num_labels
        self.rule_dimension = rule_dimension
        self.fcn1 = nn.Linear(self.rule_dimension, 8)
        self.dropout1 = nn.Dropout(0.2, inplace=False) 
        self.batchnorm1 = nn.BatchNorm1d(8) 
        self.fcn2 = nn.Linear(8, 2)
        self.relu = nn.ReLU()

    def forward(self, rules=None):
        output = self.fcn1(rules)
        output = self.batchnorm1(output)
        output = self.relu(output)
        output = self.dropout1(output)
        output = self.fcn2(output)
        return output
