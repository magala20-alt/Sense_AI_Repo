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
"""Timeline creator for MsProf operations."""
from typing import List, Dict, Tuple

from mindspore import log as logger
from mindspore.profiler.common.constant import EventConstant
from mindspore.profiler.analysis.parser.timeline_creator.base_timeline_creator import BaseTimelineCreator
from mindspore.profiler.analysis.parser.timeline_event.timeline_event_pool import TimelineEventPool
from mindspore.profiler.analysis.parser.timeline_event.flow_event import FlowStartEvent, FlowEndEvent
from mindspore.profiler.analysis.parser.timeline_event.msprof_event import (
    MsprofMetaEvent,
    MsprofCompleteEvent,
    MsprofInstantEvent,
)


class MsprofTimelineCreator(BaseTimelineCreator):
    """Create timeline event pools for MsProf operations."""

    def __init__(self):
        super().__init__()
        self.msprof_timeline_raw_data = []
        self.acl_to_npu_flow_dict: Dict[int, Dict[str, List[MsprofCompleteEvent]]] = {}

    def create(self, data: List[Dict]) -> None:
        """Create timeline event pools from MsProf timeline data."""
        if not data:
            return
        self.msprof_timeline_raw_data = data
        flow_dict, complete_event_map = self._create_base_events(data)
        self._create_acl_to_npu_flow_dict(flow_dict, complete_event_map)

    def _create_base_events(self, msprof_timeline_data: List[Dict]) -> Tuple[Dict, Dict]:
        """Create base events from MsProf timeline data."""
        flow_dict = {}
        complete_event_map = {}

        for data in msprof_timeline_data:
            pid = data.get("pid", 0)
            pool = self.event_pools.get(pid)
            if pool is None:
                pool = TimelineEventPool(pid)
                self.event_pools[pid] = pool

            if data.get("cat") == EventConstant.HOST_TO_DEVICE_FLOW_CAT:
                if data.get("ph") == EventConstant.START_FLOW:
                    event = FlowStartEvent(data)
                    flow_dict.setdefault(event.flow_id, {}).setdefault("start", event)
                elif data.get("ph") == EventConstant.END_FLOW:
                    event = FlowEndEvent(data)
                    flow_dict.setdefault(event.flow_id, {}).setdefault("end", event)
            elif data.get("ph") == EventConstant.COMPLETE_EVENT:
                event = MsprofCompleteEvent(data)
                complete_event_map[event.unique_id] = event
                pool.add_event(event)
            elif data.get("ph") == EventConstant.INSTANT_EVENT:
                event = MsprofInstantEvent(data)
                pool.add_event(event)
            elif data.get("ph") == EventConstant.META_EVENT:
                event = MsprofMetaEvent(data)
                pool.add_event(event)

        return flow_dict, complete_event_map

    def _create_acl_to_npu_flow_dict(self, flow_dict: Dict, complete_event_map: Dict) -> None:
        """Create flow events from flow dictionary."""
        for flow in flow_dict.values():
            flow_start = flow.get("start")
            flow_end = flow.get("end")
            if flow_start and flow_end:
                hardware_event = complete_event_map.get(flow_end.unique_id)
                if not hardware_event:
                    logger.warning(
                        f"Failed to find hardware event for flow end event. "
                        f"Flow ID: {flow_end.flow_id}, Unique ID: {flow_end.unique_id}"
                    )
                    continue
                (
                    self.acl_to_npu_flow_dict.setdefault(flow_start.tid, {})
                    .setdefault(str(flow_start.ts), [])
                    .append(hardware_event)
                )

    def get_acl_to_npu_flow_dict(self) -> Dict[int, Dict[str, List[MsprofCompleteEvent]]]:
        """Return the CANN to NPU flow dictionary."""
        return self.acl_to_npu_flow_dict

    def get_chrome_trace_data(self) -> List[Dict]:
        """Return the chrome trace events."""
        return self.msprof_timeline_raw_data
