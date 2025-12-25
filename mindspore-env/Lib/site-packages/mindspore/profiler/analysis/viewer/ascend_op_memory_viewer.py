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
"""Ascend op memory viewer"""
import os
import struct
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List
from abc import ABC

from mindspore.profiler.common.tlv_decoder import TLVDecoder
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.log import ProfilerLogger
from mindspore.profiler.common.constant import ProfilerActivity, FileConstant


class OpMemoryIndexEnum(Enum):
    """Op memory index defining."""

    DEVICE_ID = 0
    TID = 1
    PID = 2
    CREATE_AT = 3
    ADDR = 4
    SIZE = 5
    USED_SIZE = 6
    PEAK_SIZE = 7
    ALLOC_SIZE = 8
    USED_BY_EVENT_SZIE = 9
    EAGER_FREE_SIZE = 10
    STEAM_PTR = 11
    STREAM_ID = 12
    FROM_PERSISTENT = 13
    IS_PERSISTENT = 14
    RUN_MODE = 15
    ALLOC_TYPE = 16
    OWNER = 17


class BaseEvent(ABC):
    """Base class for all event types."""

    def __init__(self, data: Dict):
        if not isinstance(data, dict):
            raise TypeError("Input data must be dict.")
        self._origin_data = data


class OpMemoryEvent(BaseEvent):
    """Op memory event."""

    FIX_DATA_FORMAT = "<i11QI4B"
    FIX_DATA_SIZE = struct.calcsize(FIX_DATA_FORMAT)
    FREE_VALUE = 18446744073709551615  # 2^64 - 1
    NAME_KEY = 13

    def __init__(self, data: Dict):
        super().__init__(data)
        self.fix_size_data = self._origin_data[FileConstant.FIX_SIZE_DATA]

    @property
    def device_id(self):
        """Get device id."""
        return self.fix_size_data[OpMemoryIndexEnum.DEVICE_ID.value]

    @property
    def tid(self):
        """Get tid."""
        return self.fix_size_data[OpMemoryIndexEnum.TID.value]

    @property
    def pid(self):
        """Get pid."""
        return self.fix_size_data[OpMemoryIndexEnum.PID.value]

    @property
    def create_at(self):
        """Get create at."""
        return self.fix_size_data[OpMemoryIndexEnum.CREATE_AT.value]

    @property
    def addr(self):
        """Get addr."""
        return self.fix_size_data[OpMemoryIndexEnum.ADDR.value]

    @property
    def size(self):
        """Get size."""
        return self.fix_size_data[OpMemoryIndexEnum.SIZE.value]

    @property
    def used_size(self):
        """Get used size."""
        return self.fix_size_data[OpMemoryIndexEnum.USED_SIZE.value]

    @property
    def peak_size(self):
        """Get peak size."""
        return self.fix_size_data[OpMemoryIndexEnum.PEAK_SIZE.value]

    @property
    def alloc_size(self):
        """Get alloc size."""
        return self.fix_size_data[OpMemoryIndexEnum.ALLOC_SIZE.value]

    @property
    def used_by_event_size(self):
        """Get used by event size."""
        return self.fix_size_data[OpMemoryIndexEnum.USED_BY_EVENT_SZIE.value]

    @property
    def eager_free_size(self):
        """Get eager free size."""
        return self.fix_size_data[OpMemoryIndexEnum.EAGER_FREE_SIZE.value]

    @property
    def stream_ptr(self):
        """Get stream ptr."""
        return self.fix_size_data[OpMemoryIndexEnum.STEAM_PTR.value]

    @property
    def stream_id(self):
        """Get stream id."""
        return self.fix_size_data[OpMemoryIndexEnum.STREAM_ID.value]

    @property
    def from_persistent(self):
        """Get from persistent."""
        return self.fix_size_data[OpMemoryIndexEnum.FROM_PERSISTENT.value]

    @property
    def is_persistent(self):
        """Get is persistent."""
        return self.fix_size_data[OpMemoryIndexEnum.IS_PERSISTENT.value]

    @property
    def run_mode(self):
        """Get run mode."""
        return self.fix_size_data[OpMemoryIndexEnum.RUN_MODE.value]

    @property
    def alloc_type(self):
        """Get alloc type."""
        return self.fix_size_data[OpMemoryIndexEnum.ALLOC_TYPE.value]

    @property
    def owner(self):
        """Get owner."""
        return self._origin_data.get(self.NAME_KEY, "")

    @property
    def is_alloc(self):
        """Get is alloc."""
        return self.size != self.FREE_VALUE


class AscendOpMemoryViewer:
    """
    Ascend op memory viewer.
    """

    FWK_BINARY_FILE_NAME = "mindspore.memory_usage"
    OUTPUT_FILE_NAME = "operator_memory.csv"
    HEADERS = [
        "Name",
        "Size(KB)",
        "Allocation Time(us)",
        "Release Time(us)",
        "Active Release Time(us)",
        "Duration(us)",
        "Active Duration(us)",
        "Allocation Total Allocated(MB)",
        "Allocation Total Reserved(MB)",
        "Allocation Total Active(MB)",
        "Release Total Allocated(MB)",
        "Release Total Reserved(MB)",
        "Release Total Active(MB)",
        "Stream Ptr",
        "Device Type",
    ]
    DEVICE_TYPE_FMT = "NPU:{}"
    NS_TO_US = 1000
    BYTES_TO_KB = 1024
    BYTES_TO_MB = 1024 * 1024
    EMPTY_VALUE = "N/A"
    ALLOC_TIME_INDEX = 2

    def __init__(self, **kwargs):
        self._enable_profile_memory = kwargs.get("profile_memory", False)
        self._framework_path = kwargs.get("framework_path")
        self._ascend_profiler_output_path = kwargs.get("ascend_profiler_output_path")
        self._ascend_ms_dir = kwargs.get("ascend_ms_dir")
        self._activities = kwargs.get("activities")
        ProfilerLogger.init(self._ascend_ms_dir)
        self._logger = ProfilerLogger.get_instance()
        self._op_memory_events = None
        self._op_memory_data = []
        self._addr_to_event_map = defaultdict(list)

    def save(self, data: Any = None):
        """
        Save step trace time data to csv file
        """
        self._logger.info("AscendOpMemoryViewer start")

        # No frame work data is collected when no CPU is passed in activities
        if ProfilerActivity.CPU.value not in self._activities or \
                not self._enable_profile_memory:
            return

        try:
            self._read_fwk_binary_file()
            self._calculate_op_memory_data()
            self._write_data()
        except Exception as e:  # pylint: disable=W0703
            self._logger.error("Failed to save op memory data: %s", str(e), exc_info=True)
        self._logger.info("AscendOpMemoryViewer end")

    def _read_fwk_binary_file(self):
        """
        Read fwk binary file
        """
        self._logger.info("Read fwk binary file start")
        op_name_file_path = os.path.join(self._framework_path, self.FWK_BINARY_FILE_NAME)
        raw_bin_data = FileManager.read_file_content(op_name_file_path, mode="rb")
        op_memory_decode_data = TLVDecoder.decode(
            raw_bin_data, OpMemoryEvent.FIX_DATA_FORMAT, OpMemoryEvent.FIX_DATA_SIZE
        )
        self._op_memory_events = [OpMemoryEvent(data) for data in op_memory_decode_data]
        self._op_memory_events = sorted(self._op_memory_events, key=lambda x: x.create_at)
        self._logger.info("Read fwk binary file done, %d events", len(self._op_memory_events))

    def _calculate_op_memory_data(self):
        """
        Calculate op memory data
        """
        self._logger.info("Calculate op memory data start")
        if not self._op_memory_events:
            self._logger.info("No op memory events")
            return

        for event in self._op_memory_events:
            self._addr_to_event_map[event.addr].append(event)

        for _, event_list in self._addr_to_event_map.items():
            row_data_list = self._get_op_mem_row_data(event_list)
            self._op_memory_data.extend(row_data_list)

        self._op_memory_data = sorted(self._op_memory_data, key=lambda x: x[self.ALLOC_TIME_INDEX])
        self._logger.info("Calculate op memory data done")

    def _combine_alloc_and_free_event(self, alloc_event: OpMemoryEvent, free_event=None):
        """
        Combine alloc and free event
        """
        if not alloc_event:
            self._logger.error("Alloc event is None")
            return []

        return [
            alloc_event.owner,  # "Name"
            alloc_event.size / self.BYTES_TO_KB,  # "Size(KB)"
            alloc_event.create_at / self.NS_TO_US,  # "Allocation Time(us)"
            (
                self.EMPTY_VALUE
                if free_event is None
                else free_event.create_at / self.NS_TO_US
            ),  # "Release Time(us)"
            self.EMPTY_VALUE,  # "Active Release Time(us)"
            (
                self.EMPTY_VALUE
                if free_event is None
                else (free_event.create_at - alloc_event.create_at) / self.NS_TO_US
            ),  # "Duration(us)"
            self.EMPTY_VALUE,  # "Active Duration(us)"
            alloc_event.used_size
            / self.BYTES_TO_MB,  # "Allocation Total Allocated(MB)"
            alloc_event.alloc_size / self.BYTES_TO_MB,  # "Allocation Total Reserved(MB)"
            alloc_event.used_size / self.BYTES_TO_MB,  # "Allocation Total Active(MB)"
            (
                self.EMPTY_VALUE
                if free_event is None
                else free_event.used_size / self.BYTES_TO_MB
            ),  # "Release Total Allocated(MB)"
            (
                self.EMPTY_VALUE
                if free_event is None
                else free_event.alloc_size / self.BYTES_TO_MB
            ),  # "Release Total Reserved(MB)"
            (
                self.EMPTY_VALUE
                if free_event is None
                else free_event.used_size / self.BYTES_TO_MB
            ),  # "Release Total Active(MB)"
            alloc_event.stream_ptr,  # "Stream Ptr"
            self.DEVICE_TYPE_FMT.format(alloc_event.device_id),  # "Device Type"
        ]

    def _get_op_mem_row_data(self, event_list: List[OpMemoryEvent]):
        """
        Get op memory row data
        """
        res = []

        if not event_list:
            self._logger.error("Event list length is less than 1")
            return res

        start_index = 0 if event_list[0].is_alloc else 1
        alloc_event, free_event = None, None
        for event in event_list[start_index:]:
            if event.is_alloc:
                alloc_event = event
            else:
                free_event = event

            if alloc_event and free_event:
                res.append(self._combine_alloc_and_free_event(alloc_event, free_event))
                alloc_event, free_event = None, None
            elif alloc_event is None and free_event:
                self._logger.warning("Alloc event is None, but free event is not None")

        if alloc_event:
            res.append(self._combine_alloc_and_free_event(alloc_event))

        return res

    def _write_data(self):
        """
        Write data to csv file
        """
        self._logger.info("Write data to csv file start")
        save_path = os.path.join(self._ascend_profiler_output_path, self.OUTPUT_FILE_NAME)
        FileManager.create_csv_file(save_path, self._op_memory_data, self.HEADERS)
        self._logger.info("Write data to csv file done, %d rows, save path: %s", len(self._op_memory_data), save_path)
