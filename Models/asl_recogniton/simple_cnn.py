import mindspore.nn as nn


class SimpleCNN(nn.Cell):
    def __init__(self, out_channels=512):
        super().__init__()

        self.features = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=3, pad_mode="pad", padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, pad_mode="pad", padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, pad_mode="pad", padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, out_channels, kernel_size=3, pad_mode="pad", padding=1),
            nn.ReLU()
        )

    def construct(self, x):
        return self.features(x)
