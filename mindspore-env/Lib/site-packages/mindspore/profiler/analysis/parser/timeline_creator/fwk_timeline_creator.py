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
"""Timeline creator for framework operations."""
from typing import List, Dict

from mindspore.profiler.common.constant import EventConstant, FileConstant, TimelineLayerName
from mindspore.profiler.analysis.parser.timeline_event.timeline_event_pool import TimelineEventPool
from mindspore.profiler.analysis.parser.timeline_creator.base_timeline_creator import BaseTimelineCreator
from mindspore.profiler.analysis.parser.timeline_event.fwk_event import (
    FwkCompleteEvent,
    FwkInstantEvent,
    OpRangeStructField,
    FwkMetaEvent
)


class FwkTimelineCreator(BaseTimelineCreator):
    """Create timeline event pools for framework operations."""

    def create(self, data: List[Dict]) -> None:
        """Create timeline event pools from framework TLV data."""
        if not data:
            return

        pool = TimelineEventPool(EventConstant.MINDSPORE_PID)
        self.event_pools[EventConstant.MINDSPORE_PID] = pool

        self._create_base_events(pool, data)
        self._create_meta_event(pool)

    def _create_base_events(self, pool: TimelineEventPool, fwk_tlv_data: List[Dict]) -> None:
        """Create base events from framework TLV data."""
        for data in fwk_tlv_data:
            if data[FileConstant.FIX_SIZE_DATA][OpRangeStructField.START_TIME_NS.value] == 0:  # Filter abnormal data
                continue
            if (data[FileConstant.FIX_SIZE_DATA][OpRangeStructField.START_TIME_NS.value] ==
                    data[FileConstant.FIX_SIZE_DATA][OpRangeStructField.END_TIME_NS.value]):  # dur == 0
                event = FwkInstantEvent(data)
            else:
                event = FwkCompleteEvent(data)
                if event.name == EventConstant.FLOW_OP:
                    pool.add_start_event(str(event.id), event)
                    continue
                if event.id != EventConstant.INVALID_FLOW_ID:
                    pool.add_end_event(str(event.id), event)
            pool.add_event(event)

    @staticmethod
    def _create_meta_event(pool: TimelineEventPool) -> None:
        """Create meta events for framework operations."""
        process_meta_events = [
            (EventConstant.PROCESS_NAME, {"name": TimelineLayerName.MINDSPORE.value}),
            (EventConstant.PROCESS_SORT, {"sort_index": EventConstant.MINDSPORE_SORT_IDX}),
            (EventConstant.PROCESS_LABEL, {"labels": EventConstant.CPU_LABEL})
        ]

        for name, args in process_meta_events:
            pool.add_event(FwkMetaEvent({"name": name, "tid": 0, "args": args}))

        for tid in pool.get_all_tids():
            pool.add_event(FwkMetaEvent({"name": EventConstant.THREAD_NAME, "tid": tid,
                                         "args": {"name": f"Thread {tid}"}}))
            pool.add_event(FwkMetaEvent({"name": EventConstant.THREAD_SORT, "tid": tid,
                                         "args": {"sort_index": tid}}))
