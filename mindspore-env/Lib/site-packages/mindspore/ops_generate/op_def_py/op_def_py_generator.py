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
Generate Python operator definitions.
"""

import os

import common.gen_constants as K
import common.gen_utils as gen_utils

# refactored
import common.template_utils as template

from common.base_generator import BaseGenerator


class OpDefPyGenerator(BaseGenerator):
    """
    This class is responsible for generating Python operator definitions based on provided
    operation prototypes and documentation strings. It generates the code for the operator
    functions that can be used in Python scripts to interact with the underlying operations.
    """

    def __init__(self):
        """
        Initializes the generator with the template for primitive class definitions.
        """
        self.op_prim_class_define_template = template.OP_PRIM_CLASS_DEFINE_TEMPLATE

    def generate(self, work_path, op_protos, doc_dict, file_pre):
        """
        Generates Python code for operator definitions and saves it to a file.

        Args:
            work_path (str): The base directory where the generated files will be saved.
            op_protos (list): A list of operation prototypes to generate Python code for.
            doc_dict (dict): A dictionary containing documentation strings for the operators.
            file_pre (str): The prefix for the generated Python files.

        Returns:
            None

        The generated Python code includes function definitions for each operator, using
        the provided operation prototypes and documentation. It saves the code in a file
        with the given prefix in the specified work path.
        """

        gen_py = self._generate_func_code(op_protos, doc_dict)
        res_str = template.PY_LICENSE_STR + \
                  template.OPS_PY_DEF_HEADER + gen_py[:-len(template.NEW_LINE)]
        save_path = os.path.join(work_path, K.PY_AUTO_GEN_PATH)
        file_name = f"{file_pre}_ops_def.py"
        gen_utils.save_file(save_path, file_name, res_str)

    def _generate_func_code(self, op_protos, doc_dict):
        """
        Generate Python source code for operator functions based on a list of
        operator protocols and their documentation.
        """
        gen_py = "\n"
        for op_proto in op_protos:
            if op_proto.op_function.disable:
                continue
            class_name = op_proto.op_class.name
            func_name = op_proto.op_function.name
            op_args = op_proto.op_args
            func_args, prim_call_args, prim_init_args = self.get_op_args(op_args)

            func_code = "\n"
            description = gen_utils.get_op_description(op_proto.op_name, doc_dict)
            func_formal_param = ", ".join(arg_name for arg_name in func_args)
            op_prim_input_args = ", ".join(arg_name for arg_name in prim_call_args)
            if prim_init_args:
                if op_proto.op_dispatch and op_proto.op_dispatch.enable:
                    func_impl_input_args = ", ".join(op_args.arg_name for op_args in op_args)
                    func_code += f"def {func_name}({func_formal_param}):\n"
                    func_code += f"{description}"
                    func_code += f"    return {op_proto.op_name}_impl({func_impl_input_args})\n"
                else:
                    cache_prim_input_args = ", ".join(arg_name for arg_name in prim_init_args)
                    func_code += f"def {func_name}({func_formal_param}):\n"
                    func_code += f"{description}"
                    func_code += f"    {op_proto.op_name}_op = _get_cache_prim({class_name})({cache_prim_input_args})\n"
                    func_code += f"    return {op_proto.op_name}_op({op_prim_input_args})\n"
            else:
                if op_proto.op_class and op_proto.op_class.disable:
                    gen_py += f"{op_proto.op_name}_op={class_name}()\n"
                func_code += f"def {func_name}({func_formal_param}):\n"
                func_code += f"{description}"
                func_code += f"    return {op_proto.op_name}_op({op_prim_input_args})\n"

            gen_py += func_code
            gen_py += "\n"

        return gen_py

    def get_op_args(self, op_args):
        """
        Processes the list of OpArg objects to categorize them into function arguments,
        primitive initialization arguments, and primitive call arguments.

        Args:
            op_args (list): A list of OpArg objects representing the arguments of an operator.

        Returns:
            tuple: A tuple containing three lists:
                - func_args (list): Names of the function arguments.
                - prim_call_args (list): Names of the primitive call arguments.
                - prim_init_args (list): Names of the primitive initialization arguments.
        """
        func_args = []
        prim_init_args = []
        prim_call_args = []
        for op_arg in op_args:
            # step1: Process function args.
            if op_arg.default is None:
                func_args.append(f"""{op_arg.arg_name}""")
            else:
                func_args.append(f"""{op_arg.arg_name}={op_arg.default}""")

            # step2: Process primitive object init args.
            if op_arg.is_prim_init:
                prim_init_args.append(op_arg.arg_name)
            # step3: Process primitive object call args.
            else:
                prim_call_args.append(op_arg.arg_name)
        return func_args, prim_call_args, prim_init_args


class CustomOpDefPyGenerator(OpDefPyGenerator):
    """
    This class is responsible for generating Python operator definitions based on provided
    operation prototypes and documentation strings. It generates the code for the operator
    functions that can be used in Python scripts to interact with the underlying operations.
    """

    def __init__(self):
        """
        Initializes the generator with the template for primitive class definitions.
        """
        super(CustomOpDefPyGenerator).__init__()
        self.op_prim_class_define_template = template.OP_PRIM_CLASS_DEFINE_TEMPLATE

    def generate(self, work_path, op_protos, doc_dict, file_pre):
        """
        Generates Python code for operator definitions and saves it to a file.

        Args:
            work_path (str): The base directory where the generated files will be saved.
            op_protos (list): A list of operation prototypes to generate Python code for.
            doc_dict (dict): A dictionary containing documentation strings for the operators.
            file_pre (str): The prefix for the generated Python files.

        Returns:
            None

        The generated Python code includes function definitions for each operator, using
        the provided operation prototypes and documentation. It saves the code in a file
        with the given prefix in the specified work path.
        """

        gen_py = self._generate_func_code(op_protos, doc_dict)
        res_str = template.PY_LICENSE_STR + \
                  template.CUSTOM_OPS_PY_DEF_HEADER + gen_py[:-len(template.NEW_LINE)]
        file_name = f"{file_pre}_ops_def.py"
        gen_utils.save_file(work_path, file_name, res_str)
