# import mindspore.nn as nn
import tensorflow as tf
from tensorflow.keras import layers

class DualStreamWithLoss(tf.keras.Model):
    def __init__(self, network, loss_fn):
        super().__init__(auto_prefix=False)
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, rgb, pose, label):
        logits = self.network(rgb, pose)
        loss = self.loss_fn(logits, label)
        return loss
