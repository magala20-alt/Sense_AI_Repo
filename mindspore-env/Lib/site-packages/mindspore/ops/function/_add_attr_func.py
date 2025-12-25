# Copyright 2025 Huawei Technologies Co., Ltd
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
"""_add_attr"""

__all__ = ["_add_attr"]

from typing import Callable
import mindspore as ms
from mindspore._c_expression import AddAttr_
from mindspore.ops.primitive import Primitive


class AddAttr(AddAttr_):
    """
    Config primitive attributes for operators in function.
    """
    def __init__(self):
        super().__init__("AddAttr")
        self.addattr_fn = None
        self.fn = None
        self.attrs_dict = None

    def __call__(self, fn, **kwargs):
        if not isinstance(kwargs, dict):
            raise TypeError(f"the parameter 'kwargs' must be dict type, but got:{type(kwargs)}")
        if (not isinstance(fn, Callable)) or isinstance(fn, Primitive):
            raise TypeError(f"the parameter 'fn' must be callable type except Primitive, but got:{type(fn)}")

        if self._is_attr_set(fn, kwargs):
            return self.addattr_fn

        add_attr_ = AddAttr()
        attr_kv_pair = tuple(kwargs.items())

        @ms.common.jit
        def add_attr_fn(*args):
            return add_attr_(fn, attr_kv_pair)(*args)

        self.addattr_fn = add_attr_fn
        self.fn = fn
        self.attrs_dict = kwargs
        return self.addattr_fn

    def _is_attr_set(self, fn, kwargs):
        return self.addattr_fn is not None and self.addattr_fn == fn and \
               self.fn is not None and self.fn == fn and \
               self.attrs_dict is not None and self.attrs_dict == kwargs


def _add_attr(fn, **kwargs):
    return AddAttr()(fn, **kwargs)
