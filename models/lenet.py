import torch.nn as nn
import torch.nn.functional as F

class LeNet_300_100(nn.Module):
    """Simple NN with hidden layers [300, 100]

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """
    def __init__(self, save_features=None, bench_model=False, last="logsoftmax"):
        super(LeNet_300_100, self).__init__()
        self.fc1 = nn.Linear(28*28, 300, bias=True)
        self.fc2 = nn.Linear(300, 100, bias=True)
        self.fc3 = nn.Linear(100, 10, bias=True)
        self.last = last

    def forward(self, x):
        x0 = x.view(-1, 28*28)
        x1 = F.relu(self.fc1(x0))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        if self.last == "logsoftmax":
            return F.log_softmax(x3, dim=1)
        elif self.last == "logits":
            return x3
        else:
            raise ValueError("Unknown last operation")

class LeNet_5_Caffe(nn.Module):
    """LeNet-5 without padding in the first layer.
    This is based on Caffe's implementation of Lenet-5 and is slightly different
    from the vanilla LeNet-5. Note that the first layer does NOT have padding
    and therefore intermediate shapes do not match the official LeNet-5.

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, save_features=None, bench_model=False, last="logsoftmax"):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, padding=0, bias=True)
        self.conv2 = nn.Conv2d(20, 50, 5, bias=True)
        self.fc3 = nn.Linear(50 * 4 * 4, 500)
        self.fc4 = nn.Linear(500, 10)
        self.last = last

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.fc3(x.view(-1, 50 * 4 * 4)))
        if self.last == "logsoftmax":
            return F.log_softmax(self.fc4(x), dim=1)
        elif self.last == "logits":
            return self.fc4(x)
        else:
            raise ValueError("Unknown last operation")
