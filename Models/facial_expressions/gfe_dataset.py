# import os
# import re
# import numpy as np
# from mindspore.dataset import GeneratorDataset

# hex_pattern = re.compile(r"^0x[0-9a-fA-F]+$")

# LABEL_MAP = {
#     "affirmative": 0,
#     "conditional": 1,
#     "doubt_question": 2,
#     "emphasis": 3,
#     "negative": 4,
#     "relative": 5,
#     "topics": 6,
#     "wh_question": 7,
#     "yn_question": 8
# }

# class GFEDataset:
#     def __init__(self, data_dir, max_len=256,max_frames=100):
#         self.samples = []
#         self.max_len = max_len
#         self.max_frames = max_frames

#         for fname in os.listdir(data_dir):
#             if fname.endswith("_datapoints.txt"):
#                 for key in LABEL_MAP:
#                     if key in fname:
#                         self.samples.append(
#                             (os.path.join(data_dir, fname), LABEL_MAP[key])
#                         )
#                         break

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         path, label = self.samples[idx]

#         values = []
#         with open(path, "r") as f:
#             for line in f:
#                 for t in line.strip().split():
#                     if hex_pattern.match(t):
#                         values.append(int(t, 16))

#         data = np.array(values, dtype=np.float32)

#         # safety check
#         if len(data) == 0:
#             data = np.zeros(self.max_len, dtype=np.float32)

#         # normalize
#         data = data / 255.0

#         # pad / truncate
#         if len(data) > self.max_len:
#             data = data[:self.max_len]
#         else:
#             data = np.pad(data, (0, self.max_len - len(data)))

#         return data, np.int32(label)


# def create_gfe_dataset(data_dir, batch_size=8, shuffle=True):
#     ds = GeneratorDataset(
#         source=GFEDataset(data_dir),
#         column_names=["feat", "label"],
#         shuffle=shuffle
#     )
#     return ds.batch(batch_size)
