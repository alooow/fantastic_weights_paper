import torch.nn as nn
import torch.nn.functional as F

class ConvNet_CIFAR10(nn.Module):
    """
    A not-so-small convolutional neural network for CIFAR10. (Num params: 988,746)
    Based on the network from the book "Deep Learning with Python" by François Chollet, section 8.2.3.
    Differences: we added padding, because the CIFAR10 images are smaller (only 32x32).
    """
    def __init__(self, save_features=None, bench_model=False, last="logsoftmax"):
        super().__init__()

        # input shape: ([b,] 3, 32, 32) ([batch,] channels, height, width)
        # is rescaling from [0, 255] to [0, 1] necessary? Or is the input already normalized?
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # out: [b, 32, 32, 32]
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # out: [b, 32, 16, 16]
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # out: [b, 64, 16, 16]
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # out: (b, 64, 8, 8)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # out: [b, 128, 8, 8]
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # out: [b, 128, 4, 4]
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # out: [b, 256, 4, 4]
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # out: [b, 256, 2, 2]
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # out: [b, 256, 2, 2]
        self.flatten = nn.Flatten(start_dim=1)  # out: [b, 1024]
        self.fc1 = nn.Linear(256*2*2, 10)  # out: [b, 10]
        self.last = last

    def forward(self, x):
        x0 = F.relu(self.conv1(x))
        x1 = self.maxpool1(x0)
        x2 = F.relu(self.conv2(x1))
        x3 = self.maxpool2(x2)
        x4 = F.relu(self.conv3(x3))
        x5 = self.maxpool3(x4)
        x6 = F.relu(self.conv4(x5))
        x7 = self.maxpool4(x6)
        x8 = F.relu(self.conv5(x7))
        x9 = self.flatten(x8)
        if self.last == "logsoftmax":
            return F.log_softmax(self.fc1(x9), dim=1)
        elif self.last == "logits":
            return self.fc1(x9)
        else:
            raise ValueError("Unknown last operation")


class SmallConvNet_CIFAR10(nn.Module):
    """
    A small convolutional neural network for CIFAR10. Num params: 95,818. (if 4 conv layers: 390,986)
    Based on the network from the book "Deep Learning with Python" by François Chollet, section 8.2.3.
    Differences: we deleted the last 2 maxpool+conv layers and replaced them with avg pooling.
    """
    def __init__(self, save_features=None, bench_model=False, last="logsoftmax"):
        super().__init__()

        # input shape: ([b,] 3, 32, 32) ([batch,] channels, height, width)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)  # out: [b, 32, 30, 30]
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # out: [b, 32, 15, 15]
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)  # out: [b, 64, 13, 13]
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # out: [b, 64, 6, 6]
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)  # out: [b, 128, 4, 4]
        # here we do an avg pool
        self.flatten = nn.Flatten(start_dim=1)  # out: [b, 128]
        self.fc1 = nn.Linear(128, 10)  # out: [b, 10]
        self.last = last

    def forward(self, x):
        x0 = F.relu(self.conv1(x))
        x1 = self.maxpool1(x0)
        x2 = F.relu(self.conv2(x1))
        x3 = self.maxpool2(x2)
        x4 = F.relu(self.conv3(x3))
        x7 = F.avg_pool2d(x4, x4.size()[3])
        x8 = self.flatten(x7)
        if self.last == "logsoftmax":
            return F.log_softmax(self.fc1(x8), dim=1)
        elif self.last == "logits":
            return self.fc1(x8)
        else:
            raise ValueError("Unknown last operation")
