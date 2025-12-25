# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Container for managing trace view data and event pools."""
from typing import Dict, List, Optional
from collections import defaultdict

from mindspore.profiler.analysis.parser.timeline_event.timeline_event_pool import TimelineEventPool
from mindspore.profiler.analysis.parser.timeline_event.base_event import BaseEvent
from mindspore.profiler.common.constant import ProfilerStepNameConstant, TimelineLayerName


class TraceViewContainer:
    """Container for trace view data and event pools.

    This class is responsible for:
    1. Storing and managing trace event pools by process ID
    2. Maintaining process name mappings
    3. Collecting and managing trace view events
    4. Providing access to stored pools and events
    """

    def __init__(self):
        self.event_pools: Dict[int, TimelineEventPool] = {}
        self.name_to_pid: Dict[str, int] = {}
        self.pid_to_name: Dict[int, str] = {}
        self.trace_view: List[Dict] = []
        self._kernel_launch_op_dict: Dict[int, List[BaseEvent]] = defaultdict(list)
        self._hardware_op_event_dict: Dict[int, List[BaseEvent]] = defaultdict(list)

    @property
    def kernel_launch_op_event(self) -> Dict[int, List[BaseEvent]]:
        """Get all kernel launch events."""
        return self._kernel_launch_op_dict

    @property
    def hardware_op_event(self) -> Dict[int, List[BaseEvent]]:
        """Get all hardware events."""
        return self._hardware_op_event_dict

    @kernel_launch_op_event.setter
    def kernel_launch_op_event(self, value):
        self._kernel_launch_op_dict = value

    @hardware_op_event.setter
    def hardware_op_event(self, value):
        self._hardware_op_event_dict = value

    def add_event_pool(self, pool: TimelineEventPool) -> None:
        """Add event pool to container."""
        if pool.name and pool.name in self.name_to_pid:
            raise ValueError(f"Process name '{pool.name}' already exists.")
        self.event_pools[pool.pid] = pool
        if pool.name:
            self.name_to_pid[pool.name] = pool.pid
            self.pid_to_name[pool.pid] = pool.name

    def add_trace_events(self, events: List[Dict]) -> None:
        """Add trace view events."""
        self.trace_view.extend(events)

    def get_pool_by_pid(self, pid: int) -> Optional[TimelineEventPool]:
        """Get event pool by process ID."""
        return self.event_pools.get(pid)

    def get_pool_by_name(self, name: str) -> Optional[TimelineEventPool]:
        """Get event pool by process name."""
        pid = self.name_to_pid.get(name)
        return self.event_pools.get(pid) if pid is not None else None

    def get_trace_view(self) -> List[Dict]:
        """Get all trace view events."""
        return self.trace_view

    def get_all_pools(self) -> List[TimelineEventPool]:
        """Get all event pools."""
        return list(self.event_pools.values())

    def get_step_id_time_dict(self) -> Dict:
        """Get step id to time dict."""
        # Retrieve all events from the trace container for the Mindspore timeline layer

        mindspore_pool = self.get_pool_by_name(TimelineLayerName.MINDSPORE.value)
        if not mindspore_pool:
            return {}

        events = mindspore_pool.get_all_events()

        # Filter events that contain "ProfilerStep" and create a dictionary mapping (start_ts, end_ts) to step ID
        step_id_to_time_dict = dict(
            sorted(
                (
                    (event.name.split("#")[-1], (event.ts, event.dur + event.ts))
                    for event in events
                    if ProfilerStepNameConstant.PROFILER_STEP in event.name
                ),
                key=lambda item: item[1][0]
            )
        )

        return step_id_to_time_dict
