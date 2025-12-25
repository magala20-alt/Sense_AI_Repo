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

"""Module loading ops yaml."""

import os
from typing import Sequence, Union

from common.gen_utils import safe_load_yaml_from_dir
import common.gen_constants as K

from .resource_loader import ResourceLoader
from .resource_list import ResourceType


class YamlLoader(ResourceLoader):
    """
    YamlLoader is a utility class for loading yaml files.
    """

    def __init__(self, resouce_type: ResourceType, yaml_path: Union[Sequence[str], str]):
        """
        Initialize YamlLoader.

        Args:
            resouce_type (ResourceType): The type of the resource.
            yaml_path (Union[Sequence[str], str]): The path to the yaml file or directory.
        """
        self.type = resouce_type
        if isinstance(yaml_path, str):
            self.yaml_path = [yaml_path]
        else:
            self.yaml_path = yaml_path

    def load(self) -> dict:
        """
        Load yaml files.

        Returns:
            tuple[int, object]: The resource id and the yaml dict.
        """
        for yaml_path in self.yaml_path:
            if not os.path.isdir(yaml_path):
                raise ValueError(f"yaml path '{yaml_path}' not found")

        yaml_dict = {}
        for yaml_path in self.yaml_path:
            yaml_dict.update(safe_load_yaml_from_dir(yaml_path))

        return {self.type: yaml_dict}


class OpDocYamlLoader(YamlLoader):
    """
    OpDocYamlLoader is a class for loading op primitive doc yaml files.
    """

    def __init__(self):
        op_doc_yaml_path = os.path.join(K.WORK_DIR, K.MS_OP_DEF_YAML_PATH, "doc")
        super().__init__(ResourceType.OP_DOC_YAML, op_doc_yaml_path)


class CustomOpDocYamlLoader(YamlLoader):
    """
    CustomOpDocYamlLoader is a class for loading op primitive doc yaml files.
    """

    def __init__(self, doc_yaml_path):
        super().__init__(ResourceType.OP_DOC_YAML, doc_yaml_path)


class TensorMethodDocYamlLoader(YamlLoader):
    """
    TensorMethodDocYamlLoader is a class for loading tensor method doc yaml files.
    """

    def __init__(self):
        tensor_method_doc_yaml_path = os.path.join(K.WORK_DIR, K.MS_TENSOR_METHOD_DOC_YAML_PATH)
        super().__init__(ResourceType.TENSOR_METHOD_DOC_YAML, tensor_method_doc_yaml_path)


class MintFuncDocYamlLoader(YamlLoader):
    """
    MintFuncDocYamlLoader is a class for loading mint func doc yaml files.
    """

    def __init__(self):
        mint_func_doc_yaml_path = os.path.join(K.WORK_DIR, K.MS_MINT_FUNC_DOC_YAML_PATH)
        super().__init__(ResourceType.MINT_FUNC_DOC_YAML, mint_func_doc_yaml_path)
