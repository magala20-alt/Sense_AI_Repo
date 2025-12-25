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
"""Parser for MindSpore framework profiling data."""
import os
import struct
from typing import List, Dict, Any, Optional

from mindspore import log as logger
from mindspore.profiler.common.tlv_decoder import TLVDecoder
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.constant import ProfilerActivity, FileConstant, DeviceTarget
from mindspore.profiler.analysis.parser.base_parser import BaseParser
from mindspore.profiler.analysis.parser.timeline_event.fwk_event import FwkFixSizeFormat, OpRangeStructField
from mindspore.profiler.common.log import ProfilerLogger


class FrameworkParser(BaseParser):
    """Parser for MindSpore framework profiling data."""

    _OP_RANGE_FILE_NAME = "mindspore.op_range"
    _CPU_OP_TIMESTAMP_FILE_NAME = "cpu_op_execute_timestamp_{}.txt"

    def __init__(self, next_parser: Optional[BaseParser] = None, **kwargs):
        """Initialize FrameworkParser."""
        super().__init__(next_parser)
        self._rank_id = kwargs.get("rank_id")
        self._activities = kwargs.get("activities")
        self._step_list = kwargs.get("step_list")
        self._framework_path = kwargs.get("framework_path")
        self._device_target = kwargs.get("device_target")
        self._ascend_ms_dir = kwargs.get("ascend_ms_dir")
        self._op_range_path = os.path.join(
            self._framework_path,
            self._OP_RANGE_FILE_NAME
        )
        self._cpu_op_path = os.path.join(
            self._framework_path,
            self._CPU_OP_TIMESTAMP_FILE_NAME.format(self._rank_id),
        )
        ProfilerLogger.init(self._ascend_ms_dir)
        self._logger = ProfilerLogger.get_instance()

    def _parse(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse MindSpore framework profiling data.

        Args:
            data (Dict[str, Any]): Input data dictionary.

        Returns:
            Dict[str, Any]: Updated data with:
                - mindspore_op_list: List of parsed operation events
                - cpu_op_lines: List of CPU operation timestamp data
        """
        if ProfilerActivity.CPU.value not in self._activities:
            return data
        mindspore_op_list = self._parse_op_range_data()
        self._logger.info("FrameworkParser parse op range done.")
        cpu_op_lines = self._parse_cpu_op_data()
        self._logger.info("FrameworkParser parse cpu op done.")
        data.update(
            {
                "mindspore_op_list": mindspore_op_list,
                "cpu_op_lines": cpu_op_lines,
            }
        )
        return data

    def _parse_op_range_data(self) -> List[Dict]:
        """Read and decode MindSpore op_range data.

        Returns:
            List[Dict]: List of parsed MindSpore operation events.
        """
        if self._device_target == DeviceTarget.CPU.value:
            return []

        if not os.path.exists(self._op_range_path):
            logger.error("Failed to find op_range data. Skipping parse host profiler data.")
            return []

        try:
            op_range_bytes = FileManager.read_file_content(self._op_range_path, "rb")
            op_range_list = TLVDecoder.decode(
                op_range_bytes, FwkFixSizeFormat.OpRangeStruct, struct.calcsize(FwkFixSizeFormat.OpRangeStruct)
            )
            self._logger.info("FrameworkParser parse op range done, op_range_list length: %d", len(op_range_list))
            if not op_range_list:
                logger.error(
                    f"Failed to decode op_range data: empty result from file {self._op_range_path}"
                )
            return self._filter_op_range_list(op_range_list)
        except RuntimeError as err:
            logger.error(f"Failed to read file: {self._op_range_path}. Error: {str(err)}")
            return []

    def _filter_op_range_list(self, op_range_list: List[Dict]) -> List[Dict]:
        """Filter op_range_list based on step_list.

        Args:
            op_range_list (List[Dict]): List of operation events to filter.

        Returns:
            List[Dict]: Filtered list of operation events.
        """
        if not self._step_list or not isinstance(self._step_list, list):
            return op_range_list

        first_step = min(
            op[FileConstant.FIX_SIZE_DATA][OpRangeStructField.STEP.value]
            for op in op_range_list
        )
        adjusted_step_list = [step - 1 + first_step for step in self._step_list]
        return [
            op for op in op_range_list
            if op[FileConstant.FIX_SIZE_DATA][OpRangeStructField.STEP.value] in adjusted_step_list
        ]

    def _parse_cpu_op_data(self) -> List[str]:
        """Parse CPU op timestamp data.

        Returns:
            List[str]: List of CPU operation timestamp data lines.
        """
        if not os.path.exists(self._cpu_op_path):
            return []
        try:
            return FileManager.read_txt_file(self._cpu_op_path)
        except RuntimeError as err:
            logger.warning(f"Failed to read file: {self._cpu_op_path}. Error: {str(err)}")
            return []
