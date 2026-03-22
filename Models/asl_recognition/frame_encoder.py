# import mindspore.nn as nn
# from mindspore import ops
import tensorflow as tf
from .frame_encoder import FrameEncoder
from tensorflow.keras import layers


class FrameEncoder(tf.keras.Model):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    
    def call(self, x, training=False):
        # x shape: (batch, time, H, W, C)
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        h = tf.shape(x)[2]
        w = tf.shape(x)[3]
        c = tf.shape(x)[4]

        # Merge batch and time dims → (B*T, H, W, C)
        x = tf.reshape(x, (b * t, h, w, c))

        feats = self.backbone(x, training=training)  # (B*T, feature_dim)

        # Reshape back to temporal → (B, T, feature_dim)
        feats = tf.reshape(feats, (b, t, -1))

        return feats

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
