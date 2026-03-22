# import mindspore.nn as nn
# from mindspore import ops
import tensorflow as tf
from tensorflow.keras import layers
from .frame_encoder import FrameEncoder
from .temporal_cnn import TemporalCNN


class DualStreamTemporalASL(tf.keras.Model):
    def __init__(self, rgb_backbone, pose_backbone, num_classes=100):
        super().__init__()

        self.rgb_encoder = FrameEncoder(rgb_backbone)
        self.pose_encoder = FrameEncoder(pose_backbone)

        self.temporal = TemporalCNN(in_channels=512)
        self.classifier = layers.Dense(num_classes)

    def call(self, inputs):
        rgb, pose = inputs

        rgb_feat = self.rgb_encoder(rgb)
        pose_feat = self.pose_encoder(pose)

        fused = tf.concat([rgb_feat, pose_feat], axis=-1)
        temporal_feat = self.temporal(fused)

        return self.classifier(temporal_feat)