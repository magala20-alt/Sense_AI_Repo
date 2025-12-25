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
"""mind data parser"""
import os
from typing import Dict, Tuple, Optional, Any, List
from mindspore import log as logger
from mindspore.profiler.analysis.parser.base_parser import BaseParser
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.constant import ProfilerActivity
from mindspore.profiler.common.exceptions.exceptions import (
    ProfilerPathErrorException,
    ProfilerRawFileException
)
from mindspore.profiler.common.log import ProfilerLogger


class MindDataParser(BaseParser):
    """
    Parser for MindData profiling information.
    """
    _FILE_NAMES = {
        'pipeline': 'pipeline_profiling_{}.json',
        'cpu_utilization': 'minddata_cpu_utilization_{}.json',
        'device_queue': 'device_queue_profiling_{}.txt',
        'device_iterator': 'dataset_iterator_profiling_{}.txt'
    }

    def __init__(self, next_parser: Optional[BaseParser] = None, **kwargs):
        super().__init__(next_parser)
        self._device_id = kwargs.get("rank_id") if (ProfilerActivity.NPU.value in
                                                    kwargs.get("activities")) else kwargs.get("device_id")
        self._output_path = kwargs.get("framework_path")
        self._ascend_ms_dir = kwargs.get("ascend_ms_dir")
        ProfilerLogger.init(self._ascend_ms_dir)
        self._logger = ProfilerLogger.get_instance()
        self._file_paths = self._setup_file_paths()

    def _setup_file_paths(self) -> Dict[str, str]:
        return {
            key: os.path.join(self._output_path, file_name.format(self._device_id))
            for key, file_name in self._FILE_NAMES.items()
        }

    def _parse(self, data=None) -> Dict[str, Any]:
        if data is None:
            data = {}
        if not os.path.exists(self._file_paths['pipeline']):
            logger.info(
                "pipeline profiling file %s does not exist. Please check whether minddata profiling data should exist.",
                self._file_paths['pipeline'])
            return data
        op_id_info, sample_interval = self._parse_pipeline_info_dict()
        cpu_util_info = self._parse_cpu_util_info()
        device_trace_info = self._parse_device_trace()
        self._logger.info("MindDataParser parse done")
        data.update({
            "pipeline_info": (op_id_info, sample_interval),
            "cpu_util_info": cpu_util_info,
            "device_trace_info": device_trace_info,
            "device_queue_file_found": self._device_queue_file_found
        })
        return data

    def _parse_pipeline_info_dict(self) -> Tuple[Dict[str, Any], float]:
        """
        Parse the pipeline information into a dictionary.
        Returns:
            Tuple[Dict[str, Any], float]: A tuple containing the parsed op_info dictionary
                                          and sampling interval.
        Raises:
            ProfilerRawFileException: If the format of the pipeline raw file is incorrect.
        """
        pipeline_info = FileManager.read_json_file(self._file_paths['pipeline'])
        if not pipeline_info:
            raise ProfilerRawFileException('The minddata pipeline file is empty.')

        sample_interval = pipeline_info.get('sampling_interval')
        op_info = pipeline_info.get('op_info')

        if sample_interval is None or not op_info:
            raise ProfilerRawFileException('The format of minddata pipeline raw file is wrong.')

        dict_op_id_info = {
            item['op_id']: item for item in op_info if item
        }

        if len(dict_op_id_info) != len(op_info):
            raise ProfilerRawFileException('The content of minddata pipeline raw file is wrong.')

        return dict_op_id_info, sample_interval

    def _parse_cpu_util_info(self) -> Dict[str, Any]:
        try:
            cpu_util_info = FileManager.read_json_file(self._file_paths['cpu_utilization'])
            if not cpu_util_info:
                msg = f'The MindData CPU utilization file {self._file_paths["cpu_utilization"]} is empty.'
                raise RuntimeError(msg)
        except RuntimeError as err:
            cpu_util_info = {}
            logger.warning(f'Failed to read the MindData CPU utilization data. ERROR:{err}')
        return cpu_util_info

    def _parse_device_trace(self) -> List[Any]:
        """parse the device trace data"""
        try:
            self._device_trace_path, self._device_queue_file_found = self._setup_device_trace()
            device_trace = FileManager.read_txt_file(self._device_trace_path)
            if not device_trace:
                msg = f"The MindData trace profiling file {self._device_trace_path} is empty."
                raise RuntimeError(msg)
        except RuntimeError as err:
            device_trace = []
            logger.warning(f'Failed to read the MindData device trace data. ERROR:{err}')
        return device_trace

    def _setup_device_trace(self) -> Tuple[str, bool]:
        """
        Set up the device trace file path.
        Returns:
            Tuple[str, bool]: A tuple containing the device trace file path and a boolean
                              indicating whether the device queue file was found.
        Raises:
            ProfilerPathErrorException: If no device trace file is found.
        """
        queue_path = self._file_paths['device_queue']
        iterator_path = self._file_paths['device_iterator']

        if os.path.exists(queue_path):
            return queue_path, True
        if os.path.exists(iterator_path):
            return iterator_path, False

        raise ProfilerPathErrorException('A MindData device trace profiling file cannot be found.')
