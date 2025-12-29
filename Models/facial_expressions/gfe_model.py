import mindspore.nn as nn

class GFEModel(nn.Cell):
    def __init__(self, input_dim, num_classes=9):
        super().__init__()

        self.net = nn.SequentialCell(
            nn.Dense(input_dim, 512),
            nn.ReLU(),
            nn.Dense(512, 256),
            nn.ReLU(),
            nn.Dense(256, num_classes)
        )

    def construct(self, x):
        # x: (B, T, F)
        x = x.mean(axis=1)  # temporal average pooling
        return self.net(x)
