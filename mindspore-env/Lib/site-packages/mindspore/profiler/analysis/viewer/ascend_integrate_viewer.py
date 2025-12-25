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
"""Ascend integrate viewer"""
import os
import glob
from typing import List

from mindspore.profiler.analysis.viewer.base_viewer import BaseViewer
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.log import ProfilerLogger


class AscendIntegrateViewer(BaseViewer):
    """Ascend integrate viewer"""

    CSV_PREFIX_NAME = ["l2_cache", "api_statistic", "op_statistic", "static_op_mem",
                       "hbm", "nic", "roce", "pcie", "hccs"]
    AI_CPU_CSV_PATTERN = "aicpu_[0-9]*.csv"

    def __init__(self, **kwargs):
        super().__init__()
        self._output_path = kwargs.get("ascend_profiler_output_path")
        self._framework_path = kwargs.get("framework_path")
        self._analyze_json_path = kwargs.get("analyze_json_path")
        self._msprof_profile_output_path = kwargs.get("msprof_profile_output_path")
        self._ascend_ms_dir = kwargs.get("ascend_ms_dir")
        ProfilerLogger.init(self._ascend_ms_dir)
        self._logger = ProfilerLogger.get_instance()

    def save(self, data=None):
        """
        Save ascend integrate data.
        """
        self._logger.info("AscendIntegrateViewer start")
        try:
            self._copy_msprof_csv_files()
            self._copy_ai_cpu_csv_file()
        except Exception as e: # pylint: disable=W0703
            self._logger.error("Failed to save ascend integrate data, error: %s", e, exc_info=True)
        self._logger.info("AscendIntegrateViewer end")

    def _copy_csv_files(self, csv_names: List[str], source_path: str):
        """
        Copy CSV files from source path to output path.
        Args:
            csv_names (List[str]): List of CSV file name prefixes
            source_path (str): Source directory path
        """
        self._logger.info("Copy csv files start")
        for csv_name in csv_names:
            src_file = os.path.join(source_path, csv_name + "_*")
            src_file_list = glob.glob(src_file)
            if src_file_list:
                dst_file = os.path.join(self._output_path, csv_name + ".csv")
                FileManager.copy_file(src_file_list[0], dst_file)
                self._logger.info("Copy csv file %s to %s", src_file_list[0], dst_file)
        self._logger.info("Copy csv files done")

    def _copy_ai_cpu_csv_file(self):
        """
        Copy ai cpu csv files from source path to output path.
        """
        src_file = os.path.join(self._msprof_profile_output_path, self.AI_CPU_CSV_PATTERN)
        src_file_list = glob.glob(src_file)
        if src_file_list:
            dst_file = os.path.join(self._output_path, "data_preprocess.csv")
            FileManager.copy_file(src_file_list[0], dst_file)
            self._logger.info("Copy aicpu file %s to %s", src_file_list[0], dst_file)
        self._logger.info("Copy aicpu csv files done")

    def _copy_msprof_csv_files(self):
        """
        Copy msprof csv files from source path to output path.
        """
        self._copy_csv_files(self.CSV_PREFIX_NAME, self._msprof_profile_output_path)
        self._logger.info("Copy msprof csv files done")
