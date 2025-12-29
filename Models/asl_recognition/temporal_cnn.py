import mindspore.nn as nn

class TemporalCNN(nn.Cell):
    def __init__(self, in_channels=512):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size=3, padding=1, pad_mode="pad")
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def construct(self, x):
        # x: (B, T, C)
        x = x.transpose(0, 2, 1)   # (B, C, T)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        return x.squeeze(-1)       # (B, 256)
