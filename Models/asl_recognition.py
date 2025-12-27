import os
import cv2
import numpy as np
from mindspore.dataset import GeneratorDataset

# class WLASLFramesDataset:
#     def __init__(self, root_dir, max_frames=30, img_size=224):
#         self.samples = []
#         self.max_frames = max_frames
#         self.img_size = img_size
#         self.label_map = {}

#         classes = sorted(os.listdir(root_dir))

#         for label, cls in enumerate(classes):
#             class_path = os.path.join(root_dir, cls)

#             if not os.path.isdir(class_path):
#                 continue

#             self.label_map[cls] = label

#             # Each subfolder = one signer/sample
#             for signer_folder in os.listdir(class_path):
#                 signer_path = os.path.join(class_path, signer_folder)

#                 if os.path.isdir(signer_path):
#                     self.samples.append((signer_path, label))

#     def __len__(self):
#         return len(self.samples)

#     def _load_frames(self, sample_dir):
#         frame_files = sorted(os.listdir(sample_dir))
#         frames = []

#         for f in frame_files[:self.max_frames]:
#             img_path = os.path.join(sample_dir, f)
#             img = cv2.imread(img_path)
#             if img is None:
#                 continue

#             img = cv2.resize(img, (self.img_size, self.img_size))
#             img = img.astype(np.float32) / 255.0
#             frames.append(img)

#         if len(frames) == 0:
#             frames = [np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)]

#         while len(frames) < self.max_frames:
#             frames.append(frames[-1])

#         return np.array(frames, dtype=np.float32)

#     def __getitem__(self, idx):
#         path, label = self.samples[idx]
#         frames = self._load_frames(path)
#         return frames, label


# def create_dataset(root, batch_size=8, shuffle=True):
#     ds = GeneratorDataset(
#         source=WLASLFramesDataset(root),
#         column_names=["frames", "label"],
#         shuffle=shuffle
#     )
#     return ds.batch(batch_size)

# train_ds = create_dataset("preprocessing/train/frames")
# val_ds = create_dataset("preprocessing/val/frames")
# test_ds = create_dataset("preprocessing/test/frames")

class WLASLDualStreamDataset:
    def __init__(self, frames_root, pose_root, max_frames=30, img_size=224):
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

        return rgb_seq, pose_seq, label
    
def create_dual_stream_dataset(frames_root, pose_root, batch_size=4, shuffle=True):
    ds = GeneratorDataset(
        source=WLASLDualStreamDataset(frames_root, pose_root),
        column_names=["rgb", "pose", "label"],
        shuffle=shuffle
    )
    return ds.batch(batch_size)

train_ds = create_dual_stream_dataset(
    "dataset/preprocessing/train/frames",
    "dataset/preprocessing/train/pose"
)

# test data loading
print("Number of samples:", train_ds.get_dataset_size())

# inspect one batch
for batch in train_ds.create_dict_iterator():
    print("RGB shape:", batch["rgb"].shape)
    print("Pose shape:", batch["pose"].shape)
    print("Labels:", batch["label"])
    break

# visual check
import matplotlib.pyplot as plt

for batch in train_ds.create_dict_iterator():
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
    "WLASL/train/frames",
    "WLASL/train/pose"
)

print("Number of classes:", len(dataset.label_map))
print("Example label map:", list(dataset.label_map.items())[:5])




