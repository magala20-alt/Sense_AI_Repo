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
Experimental custom pass registration functionality.

This module provides experimental APIs for registering custom optimization passes.
These APIs are subject to change and should be used with caution in production code.
"""

import mindspore._c_expression as _c_expression


def register_custom_pass(pass_name: str, plugin_so_path: str, device: str = "all", stage: str = "") -> bool:
    """Register a custom pass plugin.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        pass_name (str): Name of the pass expected to be provided by the plugin.
        plugin_so_path (str): Absolute path to the plugin shared library (.so file).
        device (str): Target device for the pass. Supported values: "cpu", "gpu", "ascend", or "all".
            Default: "all".
        stage (str): Pass stage. Reserved field for future use. Default: "".

    Returns:
        bool: True if plugin loaded successfully, False otherwise.

    Examples:
        >>> import mindspore.graph as graph
        >>> # Register a custom optimization pass
        >>> success = graph.register_custom_pass(
        ...     pass_name="my_fusion_pass",
        ...     plugin_so_path="/path/to/my_plugin.so",
        ...     device="ascend"
        ... )
        >>> print(f"Registration successful: {success}")
    """
    return _c_expression.register_custom_pass(pass_name, plugin_so_path, device, stage)


__all__ = ["register_custom_pass"]
