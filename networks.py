import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist



class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.relu = nn.ReLU(dim_hidden)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

class Encoder(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super( Encoder, self).__init__()
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(dim_hidden)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class Classifier(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(Classifier, self).__init__()
        self.encoder = Encoder(dim_in, dim_hidden)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

class Discriminator(nn.Module):
    def __init__(self, dim_in):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(dim_in, 4)
        self.relu = nn.ReLU()
        self.fc2= nn.Linear(4, 1)

    def forward(self, x):
        #noise = noise.to(device)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)

class MoE(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MoE, self).__init__()
        self.classifier = Classifier(dim_in, dim_hidden, dim_out)
        self.gate = nn.Linear(dim_in, 1)

    def forward(self, x, yg):
        yl = self.classifier(x)
        a = self.gate(x)
        a = F.sigmoid(a)
        res = yl*a+yg*(1-a)
        return res, a
