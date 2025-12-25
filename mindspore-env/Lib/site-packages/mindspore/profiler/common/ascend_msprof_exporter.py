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
"""Exporter for Ascend MSPROF data."""
import os
import glob
from collections import defaultdict
from typing import Dict, List, Optional

from mindspore import log as logger
from mindspore.profiler.common.path_manager import PathManager
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.msprof_cmd_tool import MsprofCmdTool


class AscendMsprofExporter:
    """Exporter for Ascend MSPROF data."""

    _DRV_VERSION = 467473
    _DEFAULT_MODEL_ID = 4294967295
    _STEP_TRACE_FILE_PATTERN = "step_trace_*.csv"
    _STEP_TRACE_MODEL_ID_INDEX = 9
    _STEP_TRACE_ITERATION_ID_INDEX = 1

    def __init__(self, **kwargs):
        """Initialize AscendMsprofExporter."""
        self._msprof_profile_output_path = kwargs.get("msprof_profile_output_path")
        self._msprof_profile_path = kwargs.get("msprof_profile_path")
        self._step_list = kwargs.get("step_list")
        self._export_type = kwargs.get("export_type")
        self._msprof_tool = MsprofCmdTool(self._msprof_profile_path)

    def export(self) -> None:
        """Determine the export strategy and execute the appropriate export method."""
        support_all_export = self._check_drv_version()
        if self._step_list or not support_all_export:
            self._single_export(self._step_list)
        else:
            self._all_export()
        self._msprof_tool.run_ms_analyze_cmd(self._export_type)

    def _single_export(self, step_list: Optional[List[int]]) -> None:
        """Perform single export for each model and iteration.

        Args:
            step_list (Optional[List[int]]): iteration IDs.
        """
        if not step_list:
            model_iteration_dict = self._generate_step_trace()
            if not model_iteration_dict:
                return
        else:
            model_iteration_dict = {self._DEFAULT_MODEL_ID: step_list}

        for model_id, iter_list in model_iteration_dict.items():
            self._msprof_tool.run_ms_py_export_cmd(model_id, iter_list, self._export_type)

    def _all_export(self) -> None:
        """Perform all-export for all data."""
        self._msprof_tool.run_ms_export_cmd(self._export_type)

    def _check_drv_version(self) -> bool:
        """Check if the driver version supports all-export.

        Returns:
            bool: True if driver version supports all-export, False otherwise.
        """
        msprof_info = self._msprof_tool.get_msprof_info()
        status = msprof_info.get("status", 1)
        drv_version = (
            msprof_info.get("data", {}).get("version_info", {}).get("drv_version", 0)
        )

        if status != 1 and drv_version and drv_version >= self._DRV_VERSION:
            return True

        logger.warning(
            "Current driver package does not support all-export mode. "
            "Using single export mode, which may affect performance. "
            "Consider upgrading the driver package."
        )
        return False

    def _generate_step_trace(self) -> Optional[Dict[int, List[int]]]:
        """Generate step trace data.

        Returns:
            Optional[Dict[int, List[int]]]: Model IDs to iteration IDs mapping, or None if failed.
        """
        try:
            # step1: run msprof command
            self._msprof_tool.run_ms_export_cmd(self._export_type)
            step_trace_file = glob.glob(
                os.path.join(
                    self._msprof_profile_output_path, self._STEP_TRACE_FILE_PATTERN
                )
            )
            if not step_trace_file:
                logger.warning(
                    f"No step trace csv file found in {self._msprof_profile_output_path}."
                )
                return None
            # step2: parse the step trace file to get model id and iteration id
            model_iteration_dict = self._parse_step_trace_file(step_trace_file[0])
            # step3: remove the unused files
            PathManager.remove_path_safety(self._msprof_profile_output_path)
            return model_iteration_dict

        except RuntimeError as err:
            logger.warning(f"Failed to get step trace data. Error: {str(err)}")
            return None

    def _parse_step_trace_file(self, file_path: str) -> Dict[int, List[int]]:
        """Parse the step trace file to extract model and iteration information.

        Args:
            file_path (str): Path to the step trace file.

        Returns:
            Dict[int, List[int]]: Model IDs to iteration IDs mapping.
        """
        model_iteration_dict = defaultdict(list)
        step_trace_data = FileManager.read_csv_file(file_path)
        for row in step_trace_data:
            model_id = int(row[self._STEP_TRACE_MODEL_ID_INDEX])
            iteration_id = int(row[self._STEP_TRACE_ITERATION_ID_INDEX])
            model_iteration_dict[model_id].append(iteration_id)
        return model_iteration_dict
