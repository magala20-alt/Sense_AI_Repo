import mindspore.nn as nn
from mindspore import ops


class FrameEncoder(nn.Cell):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def construct(self, x):
        # x: (B, T, H, W, C)
        b, t, h, w, c = x.shape

        # (B*T, C, H, W)
        x = x.reshape(b * t, h, w, c)
        x = x.transpose(0, 3, 1, 2)

        feats = self.backbone(x)   # (B*T, C)

        # reshape back to temporal
        feats = feats.reshape(b, t, -1)  # (B, T, C)

        return feats
