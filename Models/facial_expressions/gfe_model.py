# import mindspore.nn as nn
# from mindspore import ops


# class GFEModel(nn.Cell):
#     def __init__(self, num_classes, hidden_size=128):
#         super().__init__()

#         self.expand = ops.ExpandDims()

#         self.lstm = nn.LSTM(
#             input_size=1,      # each timestep is 1 scalar
#             hidden_size=hidden_size,
#             batch_first=True
#         )

#         self.classifier = nn.Dense(hidden_size, num_classes)

#     def construct(self, x):
#         # x: (batch, seq_len)
#         x = self.expand(x, -1)        # → (batch, seq_len, 1)

#         output, _ = self.lstm(x)      # → (batch, seq_len, hidden)

#         last = output[:, -1, :]       # last timestep

#         return self.classifier(last)
