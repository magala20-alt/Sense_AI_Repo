import mindspore.nn as nn


class TemporalCNN(nn.Cell):
    def __init__(self, in_channels=1024):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 512, kernel_size=3, pad_mode="pad", padding=1)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=5, pad_mode="pad", padding=2)

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def construct(self, x):
        x = x.transpose(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x.squeeze(-1)
