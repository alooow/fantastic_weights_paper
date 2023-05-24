import torch.nn as nn
import torch.nn.functional as F

class MLP_CIFAR10(nn.Module):
    def __init__(self, save_features=None, bench_model=False, last="logsoftmax"):
        super(MLP_CIFAR10, self).__init__()

        self.fc1 = nn.Linear(3*32*32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.last = last

    def forward(self, x):
        x0 = F.relu(self.fc1(x.view(-1, 3*32*32)))
        x1 = F.relu(self.fc2(x0))
        if self.last == "logsoftmax":
            return F.log_softmax(self.fc3(x1), dim=1)
        elif self.last == "logits":
            return self.fc3(x1)
        else:
            raise ValueError("Unknown last operation")


class MLP_CIFAR10_DROPOUT(nn.Module):
    def __init__(self, density, save_features=None, bench_model=False, last="logsoftmax"):
        super(MLP_CIFAR10_DROPOUT, self).__init__()
        self.sparsity = 1-density
        self.density = density
        self.fc1 = nn.Linear(3*32*32, 1024)
        self.dropout1 = nn.Dropout(self.sparsity)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(self.sparsity)
        self.fc3 = nn.Linear(512, 10)
        self.dropout3 = nn.Dropout(self.sparsity)
        self.last = last

    def forward(self, x):
        x0 = x.view(-1, 3*32*32)
        x0 = self.dropout1(x0)
        x1 = F.relu(self.fc1(x0))
        x1 = self.dropout2(x1)
        x2 = F.relu(self.fc2(x1))
        x2 = self.dropout3(x2)
        if self.last == "logsoftmax":
            return F.log_softmax(self.fc3(x2), dim=1)
        elif self.last == "logits":
            return self.fc3(x2)
        else:
            raise ValueError("Unknown last operation")