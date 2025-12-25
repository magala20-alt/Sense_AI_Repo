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
Generates C++ helper files for primitive instance creation based on operator metadata.
"""

import os

import common.gen_constants as K
import common.gen_utils as gen_utils
import common.template as template
from common.base_generator import BaseGenerator
from pyboost import pyboost_utils


class CppCreatePrimInstanceHelperGenerator(BaseGenerator):
    """
    This class is responsible for generating a helper file that contains
    operation labels and default values for creating primitive instances in C++.
    """

    def __init__(self):
        """
        Initializes the generator with templates for operation labels and default values.
        """
        self.op_labels_template = template.op_labels_template
        self.op_args_default_value_template = template.arg_default_value
        self.op_label_template = template.Template("""    "$op_name": {$op_label_body},\n""")
        self.op_arg_default_val_template = template.Template("""    "$op_name": {$op_arg_default_value},\n""")

    def generate(self, work_path, op_protos):
        """
        Generates the content for the helper file and saves it to the specified path.

        Args:
            work_path (str): The directory where the generated file will be saved.
            op_protos (list): A list of operation prototypes to generate content for.

        Returns:
            None
        """
        py_arg_default = self.generate_op_arg_default_value(op_protos)
        py_labels = self.generate_op_labels(op_protos)
        res_str = template.PY_LICENSE_STR + py_arg_default + py_labels

        save_path = os.path.join(work_path, K.PY_AUTO_GEN_PATH)
        file_name = "cpp_create_prim_instance_helper.py"
        gen_utils.save_file(save_path, file_name, res_str)

    def generate_op_labels(self, op_protos):
        """
        Generates a string containing labels for each operation.

        Args:
            op_protos (list): A list of operation prototypes.

        Returns:
            str: A string representing the labels in the specified format.
        """
        gen_label_list = []
        for op_proto in op_protos:
            labels = op_proto.op_labels
            if labels is not None:
                op_name = pyboost_utils.get_op_name(op_proto.op_name, op_proto.op_class.name)
                op_label_list = [f"\"{name}\": {value}" for name, value in labels.items()]
                gen_label_list.append(self.op_label_template.replace(op_name=op_name, op_label_body=op_label_list))

        return self.op_labels_template.replace(gen_label_py=gen_label_list)

    def generate_op_arg_default_value(self, op_protos):
        """
        Generates a string containing default values for each operation's arguments.

        Args:
            op_protos (list): A list of operation prototypes.

        Returns:
            str: A string representing the default argument values in the specified format.
        """
        gen_default_list = []
        for op_proto in op_protos:
            arg_default_dict = {}
            for op_arg in op_proto.op_args:
                arg_default = op_arg.default
                if arg_default is not None:
                    arg_default_dict[op_arg.arg_name] = arg_default
            if arg_default_dict:
                op_name = pyboost_utils.get_op_name(op_proto.op_name, op_proto.op_class.name)
                arg_default_list = [f"\"{key}\": {value}" for key, value in arg_default_dict.items()]
                gen_default_list.append(self.op_arg_default_val_template.replace(op_name=op_name,
                                                                                 op_arg_default_value=arg_default_list))

        return self.op_args_default_value_template.replace(gen_default_py=gen_default_list)
