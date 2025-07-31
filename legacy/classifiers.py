import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Classifier1(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 64) 
        #self.input_bn = nn.BatchNorm1d(64)
        #self.input_dropout = nn.Dropout(0.50)


        self.hidden_fc1 = nn.Linear(64, 64)
        #self.bn1 = nn.BatchNorm1d(64)
        #self.dropout1 = nn.Dropout(0.50)

        # self.hidden_fc_sex = nn.Linear(64, 64)
        # # self.bn_sex = nn.BatchNorm1d(512)
        # # self.dropout_sex = nn.Dropout(0.50)

        # self.hidden_fc_ethn = nn.Linear(64, 64)
        # self.bn_ethn = nn.BatchNorm1d(512)
        # self.dropout_ethn = nn.Dropout(0.50)
         
        self.output_fc_sex = nn.Linear(64, 2)
        self.output_fc_ethnicity = nn.Linear(64, 4)
        # self.dropout = nn.Dropout(0.50)
        
        

    def forward(self, x):
        batch_size = x.shape[0]  # x = [batch size, height, width]
        x = x.view(batch_size, -1)  # x = [batch size, height * width]

        h1 = self.input_fc(x)
        #h1 = self.input_bn(h1)
        h1 = F.relu(h1)
        #h1 = self.input_dropout(h1)

        h2 = self.hidden_fc1(h1)
        #h2 = self.bn1(h2)
        h2 = F.relu(h2)
        #h2 = self.dropout1(h2)

        # h3_sex = self.hidden_fc_sex(h2)
        # #h3_sex = self.bn_sex(h3_sex)
        # h3_sex = F.relu(h3_sex)
        # #h3_sex = self.dropout_sex(h3_sex)

        # h3_ethnicity = self.hidden_fc_ethn(h2)
        # #h3_ethnicity = self.bn_ethn(h3_ethnicity)
        # h3_ethnicity = F.relu(h3_ethnicity)
        # h3_ethnicity = self.dropout_ethn(h3_ethnicity)


        y_sex= self.output_fc_sex(h2)
        y_ethnicity = self.output_fc_ethnicity(h2)
        
        return y_sex, y_ethnicity

