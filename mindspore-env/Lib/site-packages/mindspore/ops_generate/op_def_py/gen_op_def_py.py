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
Generate operator definition python files.
"""
import os
import shutil

from op_def_py.op_prim_py_generator import OpPrimPyGenerator
from op_def_py.op_def_py_generator import OpDefPyGenerator
from resources.resource_list import ResourceType
from common import gen_constants as K


def generate_ops_prim_file(work_path, op_protos, doc_dict, file_pre):
    generator = OpPrimPyGenerator()
    generator.generate(work_path, op_protos, doc_dict, file_pre)


def generate_ops_def_file(work_path, os_protos, doc_dict, file_pre):
    generator = OpDefPyGenerator()
    generator.generate(work_path, os_protos, doc_dict, file_pre)


def generate_ops_py_files(resource_mgr, file_pre='gen'):
    """
    Generate ops python file from yaml.
    """
    op_protos = resource_mgr.get_resource(ResourceType.OP_PROTO)
    op_protos += resource_mgr.get_resource(ResourceType.FUNC_OP_PROTO)
    doc_dict = resource_mgr.get_resource(ResourceType.OP_DOC_YAML)
    generate_ops_prim_file(K.WORK_DIR, op_protos, doc_dict, file_pre)
    generate_ops_def_file(K.WORK_DIR, op_protos, doc_dict, file_pre)
    shutil.copy(os.path.join(K.WORK_DIR, K.PY_OPS_GEN_PATH, 'ops_auto_generate_init.txt'),
                os.path.join(K.WORK_DIR, K.PY_AUTO_GEN_PATH, "__init__.py"))
