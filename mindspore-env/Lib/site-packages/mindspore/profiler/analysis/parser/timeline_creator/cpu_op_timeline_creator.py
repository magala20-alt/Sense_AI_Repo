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
"""Timeline creator for CPU operations."""
from typing import List

from mindspore import log as logger
from mindspore.profiler.common.constant import EventConstant
from mindspore.profiler.common.constant import TimelineLayerName
from mindspore.profiler.analysis.parser.timeline_creator.base_timeline_creator import BaseTimelineCreator
from mindspore.profiler.analysis.parser.timeline_event.timeline_event_pool import TimelineEventPool
from mindspore.profiler.analysis.parser.timeline_event.cpu_op_event import CpuOpCompleteEvent
from mindspore.profiler.analysis.parser.timeline_event.cpu_op_event import CpuOpMetaEvent


class CpuOpTimelineCreator(BaseTimelineCreator):
    """Create timeline event pools for CPU operations."""

    def __init__(self):
        super().__init__()
        self.scope_data: List[CpuOpCompleteEvent] = []

    def create(self, data: List[str]) -> None:
        """Create timeline event pools from CPU info lines."""
        if not data:
            return

        pool = TimelineEventPool(EventConstant.CPU_OP_PID)
        self.event_pools[EventConstant.CPU_OP_PID] = pool

        self._create_base_events(pool, data)
        self._create_meta_event(pool)

    def _create_base_events(self, pool: TimelineEventPool, cpu_info_lines: List[str]) -> None:
        """Create base events from CPU info lines."""
        for line in cpu_info_lines:
            line = line.strip()
            if not line:
                continue

            op_list = line.split(';')
            if len(op_list) < 3:
                logger.warning(f"Invalid CPU info format, expected at least 3 fields but got {len(op_list)}: {line}")
                continue

            op_full_name, op_type, time_info = op_list[0], op_list[1], op_list[-1]

            for time in time_info.split():
                time_parts = time.split(',')
                if len(time_parts) != 3:
                    logger.warning(f"Invalid time info format, expected 3 fields but got {len(time_parts)}: {time}")
                    continue

                start_time, dur, tid = time_parts
                event = CpuOpCompleteEvent({
                    'name': op_full_name,
                    'tid': int(tid),
                    'ts': str(start_time),
                    'dur': str(dur),
                    'args': {'type': op_type}
                })
                pool.add_event(event)

    @staticmethod
    def _create_meta_event(pool: TimelineEventPool) -> None:
        """Create meta events for CPU operations."""
        process_meta_name_and_args = [
            (EventConstant.PROCESS_NAME, {"name": TimelineLayerName.CPU_OP.value}),
            (EventConstant.PROCESS_SORT, {"sort_index": EventConstant.CPU_OP_SORT_IDX}),
            (EventConstant.PROCESS_LABEL, {"labels": EventConstant.CPU_LABEL})
        ]
        for name, args in process_meta_name_and_args:
            pool.add_event(CpuOpMetaEvent({"name": name, "tid": 0, "args": args}))

        for tid in pool.get_all_tids():
            pool.add_event(CpuOpMetaEvent({"name": EventConstant.THREAD_NAME, "tid": tid,
                                           "args": {"name": f"Thread {tid}"}}))
            pool.add_event(CpuOpMetaEvent({"name": EventConstant.THREAD_SORT, "tid": tid,
                                           "args": {"sort_index": tid}}))
