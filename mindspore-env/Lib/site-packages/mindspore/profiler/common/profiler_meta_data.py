# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
"""Profiler Meta Data"""
import os
from typing import Dict

import mindspore.communication as comm
from mindspore.profiler.common.constant import DeviceTarget
from mindspore.profiler.common.profiler_context import ProfilerContext
import mindspore.communication._comm_helper as comm_helper
from mindspore.profiler.common.file_manager import FileManager
from mindspore import log as logger


class ProfilerMetaData:
    """
        Profiler MetaData
        This class is used to handle metadata.
    """
    metadata: Dict[str, str] = {}
    MAX_META_SIZE = 100 * 1024 * 1024  # 100MB

    @classmethod
    def get_metadata(cls) -> Dict[str, str]:
        """Get metadata"""
        return cls.metadata

    @classmethod
    def set_metadata(cls, value: Dict[str, str]):
        """Set metadata"""
        cls.metadata = value

    @classmethod
    def dump_metadata(cls):
        """Dump metadata to file."""
        cls.add_group_info_to_metadata()
        if not cls.metadata:
            return
        save_path = os.path.join(ProfilerContext().ascend_ms_dir, "profiler_metadata.json")
        FileManager.create_json_file(save_path, cls.metadata)
        cls.metadata.clear()

    @classmethod
    def add_group_info_to_metadata(cls):
        """Add parallel group info to metadata"""
        try:
            if ProfilerContext().device_target == DeviceTarget.NPU.value and comm.GlobalComm.INITED \
                    and comm.GlobalComm.BACKEND == comm_helper.Backend.HCCL:
                group_info = {}
                # pylint: disable=protected-access
                for group_name in comm_helper._get_group_map().keys():
                    comm_name = comm.get_comm_name(group_name)
                    if not comm_name:
                        continue
                    group_info[comm_name] = {
                        "group_name": group_name,
                        "group_rank": comm.get_local_rank(group_name),
                        "global_ranks": comm.get_process_group_ranks(group_name)
                    }
                if group_info:
                    cls.metadata.update({"parallel_group_info": group_info})
        except Exception as err:    # pylint: disable=broad-except
            logger.error(f"Failed to get parallel group info, Exception: {str(err)}")
