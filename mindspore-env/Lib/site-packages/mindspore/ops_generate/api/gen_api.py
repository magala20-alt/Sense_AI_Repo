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
"""
Generate api definition files
"""
import copy

from resources.resource_list import ResourceType
from common import gen_constants as K
from tensor_py_cc_generator import TensorPyCppGenerator

from .tensor_func_reg_cpp_generator import TensorFuncRegCppGenerator
from .functional_map_cpp_generator import FunctionalMapCppGenerator
from .add_tensor_docs_generator import AddTensorDocsGenerator
from .functional_overload_py_generator import FunctionalOverloadPyGenerator
from .cpp_create_prim_instance_helper_generator import CppCreatePrimInstanceHelperGenerator


def gen_tensor_func_code(work_path, op_protos, func_protos, alias_api_mapping):
    generator = TensorFuncRegCppGenerator()
    generator.generate(work_path, op_protos, func_protos, alias_api_mapping)


def gen_functional_map_code(work_path, tensor_method_protos, mint_func_protos, alias_api_mapping):
    generator = FunctionalMapCppGenerator()
    generator.generate(work_path, tensor_method_protos, mint_func_protos, alias_api_mapping)


def gen_tensor_docs_code():
    generator = AddTensorDocsGenerator()
    generator.generate()


def gen_functional_overload_py(work_path, mint_func_protos, alias_api_mapping):
    generator = FunctionalOverloadPyGenerator()
    generator.generate(work_path, mint_func_protos, alias_api_mapping)


def gen_tensor_py_cc(work_path, tensor_method_protos, alias_api_mapping):
    generator = TensorPyCppGenerator()
    generator.generate(work_path, tensor_method_protos, alias_api_mapping)


def generate_create_instance_helper_file(resource_mgr):
    """
    Generate C++ helper file from yaml.
    """
    op_protos = resource_mgr.get_resource(ResourceType.OP_PROTO)
    tensor_method_protos = resource_mgr.get_resource(ResourceType.TENSOR_METHOD_PROTOS)
    op_protos_with_deprecated = get_tensor_op_protos_with_deprecated(tensor_method_protos, op_protos)
    generator = CppCreatePrimInstanceHelperGenerator()
    generator.generate(K.WORK_DIR, op_protos_with_deprecated)


def get_tensor_op_protos_with_deprecated(func_protos, op_protos):
    """
    Get op_protos with deprecated op_protos from func_protos.
    """
    tensor_op_protos = copy.deepcopy(op_protos)
    for _, item in func_protos.items():
        for func_proto in item:
            op_name = func_proto.op_proto.op_name
            if "deprecated" in func_proto.op_proto.op_name:
                func_proto.op_proto.op_class.name = ''.join(word.capitalize() for word in op_name.split('_'))
                if func_proto.op_proto.op_name[-1] == '_':
                    func_proto.op_proto.op_class.name += '_'
                tensor_op_protos.append(func_proto.op_proto)
    return tensor_op_protos


def generate_api_files(resource_mgr):
    """
    Generate api-related files.
    """
    work_path = K.WORK_DIR
    op_protos = resource_mgr.get_resource(ResourceType.OP_PROTO)
    tensor_method_protos = resource_mgr.get_resource(ResourceType.TENSOR_METHOD_PROTOS)
    mint_func_protos = resource_mgr.get_resource(ResourceType.MINT_FUNC_PROTOS)
    alias_api_mapping = resource_mgr.get_resource(ResourceType.ALIAS_API_MAPPING)
    # generate create prim instance helper file
    generate_create_instance_helper_file(resource_mgr)
    # generate tensor_py func code
    gen_tensor_func_code(work_path, op_protos, tensor_method_protos, alias_api_mapping)
    # generate functional map code
    gen_functional_map_code(work_path, tensor_method_protos, mint_func_protos, alias_api_mapping)
    # generate _tensor_docs.py that attaches docs to tensor func APIs when import mindspore
    gen_tensor_docs_code()
    # generate functional_overload.py which init pybind mint APIs from cpp
    gen_functional_overload_py(work_path, mint_func_protos, alias_api_mapping)
    # generate tensor_py.cc
    gen_tensor_py_cc(work_path, tensor_method_protos, alias_api_mapping)
