# from mindspore.vision.models import resnet18
import mindspore.nn as nn
from mindspore import Model
from mindspore.train.callback import LossMonitor, TimeMonitor

# from asl_recogniton.wlasl_dual_stream_dataset import WLASLDualStreamDataset
from asl_recogniton.wlasl_dual_stream_dataset import create_dual_stream_dataset
from asl_recogniton.dual_stream_temporal_asl import DualStreamTemporalASL
from asl_recogniton.simple_cnn import SimpleCNN


train_ds = create_dual_stream_dataset(
    "dataset/WLASL/train/frames",
    "dataset/WLASL/train/pose",
    batch_size=4
)

val_ds = create_dual_stream_dataset(
    "dataset/WLASL/val/frames",
    "dataset/WLASL/val/pose",
    batch_size=4,
    shuffle=False
)

train_ds = train_ds.project(["rgb", "pose", "label"])
val_ds   = val_ds.project(["rgb", "pose", "label"])

for batch in train_ds.create_dict_iterator():
    print(batch.keys())
    print(batch["rgb"].shape)
    print(batch["pose"].shape)
    print(batch["label"].shape)
    break


rgb_backbone = SimpleCNN(out_channels=512)
pose_backbone = SimpleCNN(out_channels=512)

model = DualStreamTemporalASL(
    rgb_backbone=rgb_backbone,
    pose_backbone=pose_backbone,
    num_classes=100
)

loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-4)

train_small_ds = train_ds.take(8).repeat(1)
val_small_ds = val_ds.take(2).repeat(1)

net = Model(
    network=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    metrics={"acc"}
)
# net = Model(
#     network=model,
#     loss_fn=loss_fn,
#     optimizer=optimizer,
#     metrics={"acc"},
#     labels_name="label"
# )


net.train(
    epoch=20,
    train_dataset=train_small_ds,
    # valid_dataset=val_small_ds,
    callbacks=[LossMonitor(), TimeMonitor()],
    dataset_sink_mode=False
)
