from mindspore.vision.models import resnet18
import mindspore.nn as nn
from mindspore import Model
from mindspore.train.callback import LossMonitor, TimeMonitor

from datasets.wlasl_dual_stream_dataset import create_dual_stream_dataset
from models.dual_stream_temporal_asl import DualStreamTemporalASL


train_ds = create_dual_stream_dataset(
    "dataset/WLASL/train/frames",
    "dataset/WLASL/train/pose",
    batch_size=4
)

rgb_backbone = resnet18(pretrained=False)
pose_backbone = resnet18(pretrained=False)

model = DualStreamTemporalASL(
    rgb_backbone=rgb_backbone,
    pose_backbone=pose_backbone,
    num_classes=100
)

loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-4)

net = Model(
    network=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    metrics={"acc"}
)

net.train(
    epoch=20,
    train_dataset=train_ds,
    callbacks=[LossMonitor(), TimeMonitor()],
    dataset_sink_mode=False
)
