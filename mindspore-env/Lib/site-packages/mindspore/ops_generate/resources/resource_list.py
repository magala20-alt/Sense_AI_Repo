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

"""Resource list."""

from enum import Enum


class ResourceType(Enum):
    OP_PROTO = 0
    DEPRECATED_OP_PROTO = 1
    OP_DOC_YAML = 2
    TENSOR_METHOD_DOC_YAML = 3
    MINT_FUNC_DOC_YAML = 4
    TENSOR_METHOD_PROTOS = 5
    MINT_FUNC_PROTOS = 6
    ALIAS_API_MAPPING = 7
    FUNC_OP_PROTO = 8
