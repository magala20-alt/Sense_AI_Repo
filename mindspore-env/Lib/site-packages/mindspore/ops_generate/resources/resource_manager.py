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

"""Module managing resource."""

from common.op_proto import OpProtoLoader, DeprecatedOpProtoLoader, FuncOpProtoLoader
from api.op_api_proto import OpApiProtoLoader

from .resource_loader import ResourceLoader
from .resource_list import ResourceType
from .yaml_loader import OpDocYamlLoader, TensorMethodDocYamlLoader, MintFuncDocYamlLoader


class ResourceManager():
    """
    ResourceManager is a class for managing resources.
    """

    def __init__(self):
        self.resource_map = {}

    def register_resource(self, loader: ResourceLoader) -> None:
        """
        Register resource.
        """
        self.resource_map.update(loader.load())

    def get_resource(self, type: ResourceType) -> object:
        """
        Get resource by type.
        """
        if type not in self.resource_map:
            raise ValueError(f"Resource '{type.name}' not registered")
        return self.resource_map[type]


def prepare_resources() -> ResourceManager:
    """
    Load needed resources.
    """
    resource_mgr = ResourceManager()
    resource_mgr.register_resource(OpProtoLoader())
    resource_mgr.register_resource(DeprecatedOpProtoLoader())
    resource_mgr.register_resource(FuncOpProtoLoader())
    resource_mgr.register_resource(OpDocYamlLoader())
    resource_mgr.register_resource(TensorMethodDocYamlLoader())
    resource_mgr.register_resource(MintFuncDocYamlLoader())
    resource_mgr.register_resource(OpApiProtoLoader(
        resource_mgr.get_resource(ResourceType.OP_PROTO),
        resource_mgr.get_resource(ResourceType.DEPRECATED_OP_PROTO),
        resource_mgr.get_resource(ResourceType.FUNC_OP_PROTO)))
    return resource_mgr
