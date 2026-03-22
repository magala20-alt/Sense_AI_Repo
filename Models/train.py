import tensorflow as tf
from asl_recognition.wlasl_dual_stream_dataset import create_dual_stream_dataset
from asl_recognition.dual_stream_temporal_asl import DualStreamTemporalASL
from asl_recognition.simple_cnn import SimpleCNN


train_ds = create_dual_stream_dataset(
    "../dataset/WLASL/train/frames",
    "../dataset/WLASL/train/pose",
    batch_size=1,
    shuffle=False
)

val_ds = create_dual_stream_dataset(
    "../dataset/WLASL/val/frames",
    "../dataset/WLASL/val/pose",
    batch_size=1,
    shuffle=False
)

for batch in train_ds.take(1):
    print(batch.keys())
    print(tf.shape(batch["rgb"]))
    print(tf.shape(batch["pose"]))
    print(tf.shape(batch["label"]))


rgb_backbone  = SimpleCNN(out_channels=256)
pose_backbone = SimpleCNN(out_channels=256)

model = DualStreamTemporalASL(
    rgb_backbone=rgb_backbone,
    pose_backbone=pose_backbone,
    num_classes=100
)

loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=["accuracy"]
)


def unpack(batch):
    return (batch["rgb"], batch["pose"]), batch["label"]

train_small_ds = train_ds.take(8).map(unpack)
val_small_ds   = val_ds.take(2).map(unpack)


model.fit(
    train_small_ds,
    epochs=2,
    callbacks=[
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch+1} — loss: {logs['loss']:.4f}")
        )
    ]
)

metrics = model.evaluate(val_small_ds, return_dict=True)
print("Validation:", metrics)


# Freeze encoders to save memory
model.rgb_encoder.trainable  = False
model.pose_encoder.trainable = False