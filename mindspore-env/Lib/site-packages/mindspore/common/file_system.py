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
"""File system registration management"""
from mindspore import log as logger
from mindspore import _checkparam as Validator

mindio_server_info = {"memfs.data_block_pool_capacity_in_gb": "100"}


class FileSystem:
    """File operation interface manager"""

    def __init__(self):
        self.create = open
        self.create_args = ("ab",)
        self.open = open
        self.open_args = ("rb",)
        self.backend = "basic"


def _register_basic_file_system(fs: FileSystem):
    """register basic file system"""
    fs.create = open
    fs.create_args = ("ab",)
    fs.open = open
    fs.open_args = ("rb",)
    return True


def _init_mindio():
    """Initialize MindIO and return the module if successful"""
    try:
        import mindio_acp as mindio
        ret = mindio.initialize(server_info=mindio_server_info)
        if ret == 0:
            return mindio
        logger.warning(f"Failed to initialize mindio_acp: ret = {ret}")
    except ImportError:
        pass
    try:
        import mindio
        ret = mindio.initialize()
        if ret == 0:
            return mindio
        logger.warning(f"Failed to initialize mindio: ret = {ret}")
    except ImportError:
        pass
    return None


def _register_mindio_file_system(fs: FileSystem):
    """register mindio file system"""
    mindio = _init_mindio()
    if mindio is None:
        return False

    fs.create = mindio.create_file
    fs.create_args = ()
    fs.open = mindio.open_file
    fs.open_args = ()
    fs.backend = "mindio"
    logger.info("The weights are stored using MindIO as the backend.")
    return True


def set_mindio_server_info(data_block_pool_capacity_in_gb=100):
    """
    Configure MindIO server settings.

    Args:
        data_block_pool_capacity_in_gb (int): Memory pool capacity for data blocks in gigabytes.
    """
    global mindio_server_info
    Validator.check_positive_int(data_block_pool_capacity_in_gb, "data_block_pool_capacity_in_gb")
    mindio_server_info["memfs.data_block_pool_capacity_in_gb"] = str(data_block_pool_capacity_in_gb)


def mindio_preload(ckpt_file_name):
    """
    Preload data into memory using MindIO for faster access.

    Args:
        ckpt_file_name (str): Checkpoint file name.

    Returns:
        bool: True if preloading is successful, False otherwise.
    """
    Validator.check_value_type('ckpt_file_name', ckpt_file_name, str, "mindio_preload")
    mindio = _init_mindio()
    if mindio is None:
        return False
    if not hasattr(mindio, 'preload'):
        logger.warning("MindIO module does not have preload method")
        return False
    mindio.preload(ckpt_file_name)
    return True
