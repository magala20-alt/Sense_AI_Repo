import os
import cv2
import numpy as np
import tensorflow as tf 
from tensorflow.keras import layers
# from tensorflow.keras.layers import Layer
from tensorflow.data import Dataset
# tf.data.Dataset.from_generator
# import tf.keras.layers.Layer
# import tf.cast

# from mindspore.dataset import GeneratorDataset
# import mindspore.nn as nn
# from mindspore import ops


class WLASLDualStreamDataset:
    def __init__(self, frames_root, pose_root, max_frames=16, img_size=112):
        self.samples = []
        self.max_frames = max_frames
        self.img_size = img_size
        self.label_map = {}

        classes = sorted(os.listdir(frames_root))

        for label, cls in enumerate(classes):
            frames_class = os.path.join(frames_root, cls)
            pose_class = os.path.join(pose_root, cls)

            if not os.path.isdir(frames_class):
                continue

            self.label_map[cls] = label

            for signer in os.listdir(frames_class):
                frames_path = os.path.join(frames_class, signer)
                pose_path = os.path.join(pose_class, signer)

                if os.path.isdir(frames_path) and os.path.isdir(pose_path):
                    self.samples.append((frames_path, pose_path, label))

    def __len__(self):
        return len(self.samples)

    def _load_sequence(self, folder):
        files = sorted(os.listdir(folder))
        frames = []

        for f in files[:self.max_frames]:
            img = cv2.imread(os.path.join(folder, f))
            if img is None:
                continue

            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            frames.append(img)

        if len(frames) == 0:
            frames = [np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)]

        while len(frames) < self.max_frames:
            frames.append(frames[-1])

        return np.array(frames, dtype=np.float32)
    
    def __getitem__(self, idx):
        frames_path, pose_path, label = self.samples[idx]

        rgb_seq = self._load_sequence(frames_path)
        pose_seq = self._load_sequence(pose_path)

        return rgb_seq, pose_seq, np.int32(label)


    # def __getitem__(self, idx):
    #     frames_path, pose_path, label = self.samples[idx]

    #     rgb_seq = self._load_sequence(frames_path)
    #     pose_seq = self._load_sequence(pose_path)

    #     return {
    #         "rgb": rgb_seq,
    #         "pose": pose_seq,
    #         "label": label
    #         }
 #rgb_seq, pose_seq, label
    
def create_dual_stream_dataset(frames_root, pose_root, batch_size=1, shuffle=True):
    dataset_obj = WLASLDualStreamDataset(frames_root, pose_root)

    def generator():
        for i in range(len(dataset_obj)):
            rgb, pose, label = dataset_obj[i]
            yield {"rgb": rgb, "pose": pose, "label": label}

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature={
            "rgb":   tf.TensorSpec(shape=(16, 112, 112, 3), dtype=tf.float32),
            "pose":  tf.TensorSpec(shape=(16, 112, 112, 3), dtype=tf.float32),
            "label": tf.TensorSpec(shape=(),               dtype=tf.int32)
        }
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataset_obj))

    return ds.batch(batch_size)


train_ds = create_dual_stream_dataset(
    "../dataset/WLASL/train/frames",
    "../dataset/WLASL/train/pose"
)

# test data loading
print("Number of samples:", tf.data.experimental.cardinality(train_ds))

# inspect one batch
for batch in train_ds.take(1):
    print("RGB shape:", tf.shape(batch["rgb"]) )
    print("Pose shape:", tf.shape(batch["pose"]))
    print("Labels:", tf.shape(batch["label"]))
    break

# visual check
import matplotlib.pyplot as plt

for batch in train_ds.take(1):
    rgb = batch["rgb"][0]
    pose = batch["pose"][0]

    plt.subplot(1, 2, 1)
    plt.imshow(rgb[0])
    plt.title("RGB Frame")

    plt.subplot(1, 2, 2)
    plt.imshow(pose[0])
    plt.title("Pose Frame")

    plt.show()
    break

# confirm class label mapping
dataset = WLASLDualStreamDataset(
    "../dataset/WLASL/train/frames",
    "../dataset/WLASL/train/pose"
)

print("Number of classes:", len(dataset.label_map))
print("Example label map:", list(dataset.label_map.items())[:5])




