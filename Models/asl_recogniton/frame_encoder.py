import mindspore.nn as nn


class FrameEncoder(nn.Cell):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) #nn.GlobalAvgPooling()

    def construct(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B * T, H, W, C)
        x = x.transpose(0, 3, 1, 2)

        feats = self.backbone(x)
        feats = self.pool(feats)

        return feats.view(B, T, -1)
