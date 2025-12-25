# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Layer.

The high-level components(Cells) used to construct the neural network.
"""
from __future__ import absolute_import

from mindspore.mint.nn.layer import normalization
from mindspore.mint.nn.layer import activation
from mindspore.mint.nn.layer import pooling
from mindspore.mint.nn.layer.normalization import GroupNorm
from mindspore.mint.nn.layer.normalization import BatchNorm1d
from mindspore.mint.nn.layer.normalization import BatchNorm2d
from mindspore.mint.nn.layer.normalization import BatchNorm3d
from mindspore.mint.nn.layer.normalization import LayerNorm
from mindspore.mint.nn.layer.normalization import SyncBatchNorm
from mindspore.mint.nn.layer.activation import LogSigmoid
from mindspore.mint.nn.layer.activation import SiLU
from mindspore.mint.nn.layer.activation import Threshold
from mindspore.mint.nn.layer.pooling import AdaptiveMaxPool1d
from mindspore.mint.nn.layer.pooling import AdaptiveAvgPool1d
from mindspore.mint.nn.layer.pooling import AdaptiveAvgPool2d
from mindspore.mint.nn.layer.pooling import AdaptiveAvgPool3d


__all__ = [
    'GroupNorm',
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d',
    'LayerNorm',
    'LogSigmoid',
    'SiLU',
    'AdaptiveMaxPool1d',
    'AdaptiveAvgPool1d',
    'AdaptiveAvgPool2d',
    'AdaptiveAvgPool3d',
    'SyncBatchNorm',
    'Threshold',
]
