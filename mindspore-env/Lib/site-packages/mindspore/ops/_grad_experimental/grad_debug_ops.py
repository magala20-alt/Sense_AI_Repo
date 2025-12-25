# Copyright 2020 Huawei Technologies Co., Ltd
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

"""Generate bprop for debug ops"""

import mindspore.ops.functional as F
from mindspore.ops import operations as P
from mindspore.ops.composite import multitype_ops as C
from mindspore.ops._grad_experimental.grad_base import bprop_getters, bprops

# Unused parameters are placeholders.


@bprop_getters.register(P.InsertGradientOf)
def get_bprop_insert_gradient_of(self):
    """Generate bprop for InsertGradientOf"""
    f = self.f

    def bprop(x, out, dout):
        fdout = f(dout)
        if fdout is None:
            dout = F.depend(dout, fdout)
            return (dout,)
        return (fdout,)
    return bprop


@bprops.register("TensorDump")
def bprop_tensor_dump(file, input_x, out, dout):
    """Generate bprop for TensorDump"""
    return file, C.zeros_like(input_x)


@bprop_getters.register(P.DumpGradient)
def get_bprop_dump_gradient(self):
    """Generate bprop for DumpGradient"""
    td = P.TensorDump()
    td.add_prim_attr("side_effect_io", False)
    td.add_prim_attr("td_flag", True)

    def bprop(path, x, input_output, out, dout):
        tded = td(path, dout)
        fdout = F.depend(dout, tded)
        return C.zeros_like(path), fdout, C.zeros_like(input_output)
    return bprop
