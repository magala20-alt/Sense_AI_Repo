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
"""ms dataset viewer"""
import os
from collections import defaultdict
from typing import List, Dict, Any

from mindspore import log as logger
from mindspore.profiler.common.constant import FileConstant
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.analysis.viewer.base_viewer import BaseViewer
from mindspore.profiler.analysis.parser.timeline_event.fwk_event import (
    FwkCompleteEvent,
    OpRangeStructField,
)
from mindspore.profiler.common.log import ProfilerLogger


class MsDatasetViewer(BaseViewer):
    """Viewer for MindSpore dataset profiling data."""

    _DATASET_FILE_NAME = 'dataset.csv'
    _DATASET_OP_PREFIX = 'Dataset'
    _COL_NAMES = ['Operation', 'Stage', 'Occurrences', 'Avg. time (us)', 'Custom Info']

    def __init__(self, **kwargs):
        super().__init__()
        self._save_path = os.path.join(
            kwargs.get("ascend_profiler_output_path"),
            self._DATASET_FILE_NAME
        )
        self._ascend_ms_dir = kwargs.get("ascend_ms_dir")
        ProfilerLogger.init(self._ascend_ms_dir)
        self._logger = ProfilerLogger.get_instance()

    def save(self, data: Dict[str, Any]) -> None:
        """Process and save dataset profiling data."""
        self._logger.info("MsDatasetViewer start")
        try:
            op_range_list = data.get("mindspore_op_list", [])
            dataset_statistics = self._calculate_data(op_range_list)
            self._save_data(dataset_statistics)
        except Exception as e: # pylint: disable=W0703
            self._logger.error("Failed to save dataset.csv: %s", e, exc_info=True)
        self._logger.info("MsDatasetViewer end")

    def _save_data(self, dataset_statistics: List[List[Any]]) -> None:
        """Save dataset statistics to a CSV file."""
        if not dataset_statistics:
            return
        self._logger.info("Save dataset statistics start")
        FileManager.create_csv_file(self._save_path, dataset_statistics, self._COL_NAMES)
        self._logger.info("Save dataset statistics done")

    def _calculate_data(self, fwk_tlv_data: List[Dict]) -> List[List[Any]]:
        """Calculate statistics for dataset operations."""
        dataset_op_data = []
        for data in fwk_tlv_data:
            if (data[FileConstant.FIX_SIZE_DATA][OpRangeStructField.START_TIME_NS.value] <
                    data[FileConstant.FIX_SIZE_DATA][OpRangeStructField.END_TIME_NS.value]):  # dur > 0
                name = data.get(OpRangeStructField.MODULE_GRAPH.value, "")
                if name == self._DATASET_OP_PREFIX:
                    dataset_op_data.append(FwkCompleteEvent(data))

        dataset_op_stats = defaultdict(list)
        for op_data in dataset_op_data:
            op_name_list = op_data.name.split('::')
            if len(op_name_list) != 3:
                logger.warning(f"Invalid dataset op name: {op_data.name}")
                continue
            _, event, stage = op_data.name.split('::')
            key = f"{event}::{stage}::{op_data.custom_info}"
            dataset_op_stats[key].append(op_data.dur)

        dataset_statistics = []
        for key, durations in dataset_op_stats.items():
            event, stage, custom_info = key.split('::')
            occurrence_count = len(durations)
            if occurrence_count == 0:
                continue
            average_duration = round(float(sum(durations) / occurrence_count), 2)
            dataset_statistics.append([event, stage, occurrence_count, average_duration, custom_info])

        return dataset_statistics
