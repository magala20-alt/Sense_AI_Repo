# import mindspore.nn as nn


# class SimpleCNN(nn.Cell):
#     def __init__(self, out_channels=256):
#         super().__init__()

#         self.features = nn.SequentialCell(
#             nn.Conv2d(3, 64, kernel_size=3, pad_mode="pad", padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(64, 128, kernel_size=3, pad_mode="pad", padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(128, 256, kernel_size=3, pad_mode="pad", padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(256, out_channels, kernel_size=3, pad_mode="pad", padding=1),
#             nn.ReLU()
#         )

#     def construct(self, x):
#         return self.features(x)

import tensorflow as tf
from tensorflow.keras import layers

class SimpleCNN(tf.keras.Model):
    def __init__(self, out_channels=256):
        super().__init__()

        self.features = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=2),

            layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=2),

            layers.Conv2D(out_channels, kernel_size=3, padding='same', activation='relu'),

            # global pooling
            layers.GlobalAveragePooling2D()
        ])

    def call(self, x):
        # x: (B*T, H, W, 3) — TF uses channels-last by default
        return self.features(x)  # (B*T, out_channels)