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
"""Ascend Step Trace Time Viewer"""
import os
import re
from decimal import Decimal
from enum import Enum
from typing import List, Any, Tuple, Optional

import numpy as np

from mindspore import log as logger
from mindspore.profiler.analysis.viewer.base_viewer import BaseViewer
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.constant import (
    TimelineLayerName,
    OverlapAnalysisTidName,
    ProfilerLevel,
    ProfilerActivity
)
from mindspore.profiler.analysis.parser.timeline_event.msprof_event import (
    MsprofCompleteEvent,
)
from mindspore.profiler.analysis.parser.timeline_event.timeline_event_pool import (
    TimelineEventPool,
)
from mindspore.profiler.analysis.parser.timeline_assembly_factory.trace_view_container import (
    TraceViewContainer,
)
from mindspore.profiler.common.log import ProfilerLogger


class StepTraceTimeHeaders(Enum):
    """Step trace time headers"""
    STEP = "Step"
    COMPUTING = "Computing"
    COMMUNICATION_NOT_OVERLAPPED = "Communication(Not Overlapped)"
    OVERLAPPED = "Overlapped"
    COMMUNICATION = "Communication"
    FREE = "Free"
    STAGE = "Stage"
    BUBBLE = "Bubble"
    COMMUNICATION_NOT_OVERLAPPED_EXCLUDE_RECEIVE = "Communication(Not Overlapped and Exclude Receive)"
    PREPARING = "Preparing"


class AscendStepTraceTimeViewer(BaseViewer):
    """Ascend Step Trace Time Viewer"""

    STEP_TRACE_TIME_FILE_NAME = "step_trace_time.csv"
    STEP_TRACE_TIME_HEADERS = [header.value for header in StepTraceTimeHeaders]

    # HCCL Send, Recv op pattern
    PP_OP_PATTERN = (
        # eg: hcom_BatchSendRecv__101_0_1
        re.compile(r"^hcom_\w+SendRecv__\d+"),
        # eg: hcom_send__101_0_1
        re.compile(r"hcom_send__\d+"),
        # eg: hcom_receive__101_0_1
        re.compile(r"hcom_receive__\d+"),
        re.compile(r"Receive-op"),
        re.compile(r"Send-op"),
    )

    # numpy array dtype
    OVERLAP_DTYPE = np.dtype([("ts", object), ("dur", object)])
    HCCL_DTYPE = np.dtype([("name", object), ("ts", object), ("dur", object)])

    def __init__(self, **kwargs):
        super().__init__()
        self._save_path = os.path.join(
            kwargs.get("ascend_profiler_output_path"), self.STEP_TRACE_TIME_FILE_NAME
        )
        self._profiler_level = kwargs.get("profiler_level")
        self._ascend_ms_dir = kwargs.get("ascend_ms_dir")
        self._activities = kwargs.get("activities")
        ProfilerLogger.init(self._ascend_ms_dir)
        self._logger = ProfilerLogger.get_instance()
        self.step_trace_time_data_list = []
        self.trace_container: TraceViewContainer = None
        self.hccl_pool: TimelineEventPool = None
        self.overlap_pool: TimelineEventPool = None
        # HCCL events
        self.hccl_events: List[MsprofCompleteEvent] = None
        # Overlap analysis events
        self.computing_events: List[MsprofCompleteEvent] = None
        self.communication_events: List[MsprofCompleteEvent] = None
        self.communication_not_overlapped_events: List[MsprofCompleteEvent] = None
        self.free_events: List[MsprofCompleteEvent] = None
        # Overlap analysis numpy array
        self.computing_np: np.ndarray = None
        self.communication_np: np.ndarray = None
        self.communication_not_overlapped_np: np.ndarray = None
        self.free_np: np.ndarray = None
        # HCCL numpy array
        self.hccl_events_np: np.ndarray = None

    def save(self, data: Any):
        """
        Save step trace time data to csv file
        """
        self._logger.info("AscendStepTraceTimeViewer start")
        if self._profiler_level == ProfilerLevel.LevelNone.value:
            return
        try:
            self._check_input_data(data)
            self._convert_events_to_numpy()
            self._calculate_step_trace_time()
            self._write_data()
        except Exception as e:  # pylint: disable=W0703
            self._logger.error("Failed to save step trace time data, error: %s", str(e), exc_info=True)
        self._logger.info("AscendStepTraceTimeViewer end")

    def _write_data(self):
        """
        Write step trace time data to csv file
        """
        self._logger.info("Write step trace time data start")
        data = [[str(item.get(header, "")) for header in self.STEP_TRACE_TIME_HEADERS]
                for item in self.step_trace_time_data_list]
        FileManager.create_csv_file(
            self._save_path,
            data,
            self.STEP_TRACE_TIME_HEADERS,
        )
        self._logger.info("Write step trace time data done, %d rows saved, save path: %s", len(data), self._save_path)

    def _check_input_data(self, data: Any):
        """
        Check input data and initialize data
        """
        self.trace_container: TraceViewContainer = data.get(
            "trace_view_container", None
        )

        if self.trace_container is None:
            raise ValueError("trace is empty")

        self.overlap_pool: TimelineEventPool = self.trace_container.get_pool_by_name(
            TimelineLayerName.OVERLAP_ANALYSIS.value
        )
        self.hccl_pool: TimelineEventPool = self.trace_container.get_pool_by_name(
            TimelineLayerName.HCCL.value
        )

        if self.overlap_pool is None:
            raise ValueError("overlap pool is empty")

        self.computing_events: List[MsprofCompleteEvent] = (
            self.overlap_pool.get_events_by_name(OverlapAnalysisTidName.COMPUTING.value)
        )
        self.communication_events: List[MsprofCompleteEvent] = (
            self.overlap_pool.get_events_by_name(
                OverlapAnalysisTidName.COMMUNICATION.value
            )
        )
        self.communication_not_overlapped_events: List[MsprofCompleteEvent] = (
            self.overlap_pool.get_events_by_name(
                OverlapAnalysisTidName.COMMUNICATION_NOT_OVERLAP.value
            )
        )
        self.free_events: List[MsprofCompleteEvent] = (
            self.overlap_pool.get_events_by_name(OverlapAnalysisTidName.FREE.value)
        )
        if self.hccl_pool is not None:
            self.hccl_events: List[MsprofCompleteEvent] = (
                self.hccl_pool.get_complete_events()
            )

    def _convert_overlap_events_to_numpy(
            self, events: List[MsprofCompleteEvent], dtype
    ):
        """
        Convert overlap events to numpy array
        """
        return np.array([(event.ts, event.dur) for event in events], dtype=dtype)

    def _convert_events_to_numpy(self):
        """
        Convert events to numpy array
        """
        self.computing_np = self._convert_overlap_events_to_numpy(
            self.computing_events, self.OVERLAP_DTYPE
        )
        self.communication_np = self._convert_overlap_events_to_numpy(
            self.communication_events, self.OVERLAP_DTYPE
        )
        self.communication_not_overlapped_np = self._convert_overlap_events_to_numpy(
            self.communication_not_overlapped_events, self.OVERLAP_DTYPE
        )
        self.free_np = self._convert_overlap_events_to_numpy(
            self.free_events, self.OVERLAP_DTYPE
        )
        self.computing_np = np.sort(self.computing_np, order="ts")
        self.communication_np = np.sort(self.communication_np, order="ts")
        self.communication_not_overlapped_np = np.sort(
            self.communication_not_overlapped_np, order="ts"
        )
        self.free_np = np.sort(self.free_np, order="ts")

        if self.hccl_events is not None:
            self.hccl_events_np = np.array(
                [(event.name, event.ts, event.dur) for event in self.hccl_events],
                dtype=self.HCCL_DTYPE,
            )
            self.hccl_events_np = np.sort(self.hccl_events_np, order="ts")

    def _calculate_step_trace_time(self):
        """
        Calculate step trace time data
        """
        step_id_to_time_dict = self._init_step_dict()
        self.generate_step_trace_time_data(step_id_to_time_dict)

    def _init_step_dict(self):
        """
        Init step list.
        """
        return self.trace_container.get_step_id_time_dict() or {0: (Decimal('0'), Decimal('Infinity'))}

    def generate_step_trace_time_data(self, step_id_to_time_dict):
        """
        Generate step trace time data
        """
        for step_id, (start_time, end_time) in step_id_to_time_dict.items():
            # step id、computing time、communication time、communication not overlapped time、free time
            computing_time = self._calculate_event_total_time_by_step(self.computing_np, start_time, end_time)
            communication_time = self._calculate_event_total_time_by_step(self.communication_np, start_time, end_time)
            communication_not_over_lapped_time = self._calculate_event_total_time_by_step(
                self.communication_not_overlapped_np, start_time, end_time)
            free_time = self._calculate_free_event_total_time_by_step(self.free_np, start_time, end_time)
            step_trace_time_data = {StepTraceTimeHeaders.STEP.value: step_id,
                                    StepTraceTimeHeaders.COMPUTING.value: computing_time,
                                    StepTraceTimeHeaders.COMMUNICATION.value: communication_time,
                                    StepTraceTimeHeaders.COMMUNICATION_NOT_OVERLAPPED.value:
                                        communication_not_over_lapped_time,
                                    StepTraceTimeHeaders.FREE.value: free_time}
            # overlapped time
            step_trace_time_data[StepTraceTimeHeaders.OVERLAPPED.value] = (
                step_trace_time_data[StepTraceTimeHeaders.COMMUNICATION.value]
                - step_trace_time_data[StepTraceTimeHeaders.COMMUNICATION_NOT_OVERLAPPED.value]
            )
            # stage time && bubble time
            (
                step_trace_time_data[StepTraceTimeHeaders.STAGE.value],
                step_trace_time_data[StepTraceTimeHeaders.BUBBLE.value],
            ) = self._calculate_stage_bubble(start_time, end_time)
            # communication not overlapped time exclude receive
            step_trace_time_data[StepTraceTimeHeaders.COMMUNICATION_NOT_OVERLAPPED_EXCLUDE_RECEIVE.value] = (
                step_trace_time_data[StepTraceTimeHeaders.COMMUNICATION_NOT_OVERLAPPED.value]
                - step_trace_time_data.get(StepTraceTimeHeaders.BUBBLE.value, Decimal('0.000'))
            )
            step_trace_time_data[StepTraceTimeHeaders.PREPARING.value] = self._calculate_prepare_time_by_step(
                self.computing_np, self.communication_np, start_time, step_id
            )
            self.step_trace_time_data_list.append(step_trace_time_data)

    def _calculate_event_total_time_by_step(self, times: np.ndarray, ts: Decimal, es: Decimal) -> Decimal:
        """
        Calculate event total time by step.
        """

        ts_values = times['ts']

        mask = (ts_values >= ts) & (ts_values <= es)
        filtered_times = times[mask]

        return Decimal(str(filtered_times['dur'].sum())).quantize(Decimal('0.000'))

    def _calculate_free_event_total_time_by_step(self, times: np.ndarray, ts: Decimal, es: Decimal) -> Decimal:
        """
        Calculate free event total time by step, with clipping of events that exceed the time range.
        """
        start_times = times['ts']
        durations = times['dur']
        end_times = start_times + durations

        # Clip start times to ts and end times to es
        clipped_start_times = np.maximum(start_times, ts)
        clipped_end_times = np.minimum(end_times, es)

        # Calculate the clipped durations
        clipped_durations = np.maximum(clipped_end_times - clipped_start_times, Decimal('0.000'))

        return Decimal(sum(clipped_durations)).quantize(Decimal('0.000'))

    def _calculate_event_first_time_by_step(self, times: np.ndarray, ts: Decimal) -> Optional[Decimal]:
        """
        Calculate event first time by step.
        """

        idx = np.searchsorted(times['ts'], ts)

        if idx >= len(times):
            return None

        return Decimal(str(times['ts'][idx])).quantize(Decimal('0.000'))

    def _calculate_prepare_time_by_step(self, computing_np: np.ndarray, communication_np: np.ndarray,
                                        ts: Decimal, step_id: int) -> Decimal:
        """
        calculate prepare time
        """

        # No frame work data is collected when no CPU is passed in activities
        if ProfilerActivity.CPU.value not in self._activities:
            return Decimal('0.000')

        step_computing_first_time = self._calculate_event_first_time_by_step(computing_np, ts)
        step_communication_first_time = self._calculate_event_first_time_by_step(communication_np, ts)

        if step_computing_first_time and step_communication_first_time:
            step_first_device_task_time = min(step_computing_first_time, step_communication_first_time)
        else:
            step_first_device_task_time = step_computing_first_time or step_communication_first_time

        if step_first_device_task_time:
            if ts == Decimal("0"):  # When Profiler.step() is not used
                fmk_api_events = self.trace_container.get_pool_by_name(
                    TimelineLayerName.MINDSPORE.value
                ).get_complete_events()
                step_host_start_time = min(event.ts for event in fmk_api_events)
            else:
                step_host_start_time = ts
            step_prepare_time = step_first_device_task_time - step_host_start_time
            return step_prepare_time.quantize(Decimal('0.000'))

        logger.warning(f"Failed to find device task in step {step_id}, set prepare time to 0")
        return Decimal('0.000')

    def _calculate_stage_bubble(self, ts: Decimal, es: Decimal) -> Tuple[Decimal, Decimal]:
        """
        Calculate stage and bubble time
        """
        if self.hccl_events is None:
            logger.info("HCCL events is empty, skip calculate stage and bubble")
            return Decimal(0), Decimal(0)

        mask = (self.hccl_events_np["ts"] >= ts) & (self.hccl_events_np["ts"] <= es)
        filtered_hccl_events_np = self.hccl_events_np[mask]

        if filtered_hccl_events_np.size == 0:
            logger.info("No HCCL events in the given time range, skip calculate stage and bubble")
            return Decimal(0), Decimal(0)

        total_hccl_time = filtered_hccl_events_np["ts"][-1] - filtered_hccl_events_np["ts"][0] + \
                          filtered_hccl_events_np["dur"][-1]
        bubble_time = np.sum(
            filtered_hccl_events_np["dur"][
                np.array(
                    [
                        self._is_send_recv_op(name)
                        for name in filtered_hccl_events_np["name"]
                    ]
                )
            ]
        )
        stage_time = total_hccl_time - bubble_time
        return stage_time, bubble_time

    def _is_send_recv_op(self, op_name: str) -> bool:
        """
        Check if the op is a send or recv op
        """
        return any(pattern.match(op_name) for pattern in self.PP_OP_PATTERN)
