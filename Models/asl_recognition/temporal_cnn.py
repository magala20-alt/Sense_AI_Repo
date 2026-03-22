import tensorflow as tf
from tensorflow.keras import layers

class TemporalCNN(tf.keras.Model):
    def __init__(self, in_channels=512):
        super().__init__()
        # Conv1D in TF expects (B, T, C), kernel_size=3, 256 filters
        self.conv1 = layers.Conv1D(filters=256, kernel_size=3, padding='same', activation=None)
        self.relu = layers.ReLU()
        self.pool = layers.GlobalAveragePooling1D()

    def call(self, x):
        # x: (B, T, C) — TensorFlow Conv1D expects this format natively
        x = self.conv1(x)   # (B, T, 256)
        x = self.relu(x)    # (B, T, 256)
        x = self.pool(x)    # (B, 256)
        return x