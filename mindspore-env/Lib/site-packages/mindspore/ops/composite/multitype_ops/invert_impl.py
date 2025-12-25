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

"""Implementation for internal polymorphism `invert` operations."""
from mindspore.ops.composite import base
from mindspore.ops import functional as F


invert = base.MultitypeFuncGraph('invert', True)

@invert.register("Number")
def _invert_scalar(x):
    """
    return the inverted value of x.

    Args:
        x (Number): x
    Returns:
        Number. Equal to -x-1, has the same type of x.
    """
    if isinstance(x, bool):
        if x:
            return -2
        return -1
    if not isinstance(x, int):
        raise TypeError(f"bad operand type for unary ~: '{type(x)}")
    return -x-1

@invert.register("Tensor")
def _invert_tensor(x):
    """
    return the inverted tensor of x.
    Arg:
        x (Tensor): x
    Returns:
        Tensor.
    """
    return F.logical_not(x)
