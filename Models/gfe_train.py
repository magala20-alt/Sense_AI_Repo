from mindspore import Model
import mindspore.nn as nn
from mindspore.train.callback import LossMonitor

from facial_expressions.gfe_dataset import create_gfe_dataset
from facial_expressions.gfe_model import GFEModel

ds = create_gfe_dataset("dataset/grammatical_facial_expressions/grammatical_facial_expression", batch_size=8)

model = GFEModel(num_classes=9)

loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
opt = nn.Adam(model.trainable_params(), 1e-3)

net = Model(model, loss_fn=loss_fn, optimizer=opt, metrics={"accuracy"})

net.train(10, ds, callbacks=[LossMonitor()], dataset_sink_mode=False)
