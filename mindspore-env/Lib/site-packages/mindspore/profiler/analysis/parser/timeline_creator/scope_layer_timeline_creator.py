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
"""Timeline creator for scope layer operations."""
from decimal import Decimal
from typing import List, Tuple, Optional

from mindspore.profiler.common.constant import EventConstant, TimelineLayerName
from mindspore.profiler.analysis.parser.timeline_event.fwk_event import FwkCompleteEvent
from mindspore.profiler.analysis.parser.timeline_event.cpu_op_event import CpuOpCompleteEvent
from mindspore.profiler.analysis.parser.timeline_event.base_event import BaseEvent, CompleteEvent
from mindspore.profiler.analysis.parser.timeline_event.timeline_event_pool import TimelineEventPool
from mindspore.profiler.analysis.parser.timeline_creator.base_timeline_creator import BaseTimelineCreator
from mindspore.profiler.analysis.parser.timeline_event.msprof_event import MsprofCompleteEvent
from mindspore.profiler.analysis.parser.timeline_event.scope_layer_event import (
    ScopeLayerCompleteEvent,
    ScopeLayerMetaEvent
)


class ScopeLayerTimelineCreator(BaseTimelineCreator):
    """Create timeline event pools for scope layer operations."""

    def create(self, data: List[BaseEvent]) -> None:
        """Create timeline event pools from scope layer events."""
        if not data:
            return

        pool = TimelineEventPool(EventConstant.SCOPE_LAYER_PID)
        self.event_pools[EventConstant.SCOPE_LAYER_PID] = pool

        self._create_base_events(pool, data)
        self._create_meta_event(pool)

    def _create_base_events(self, pool: TimelineEventPool, event_list: List[BaseEvent]) -> None:
        """Create base events from scope layer events."""
        event_list.sort(key=lambda x: x.ts)
        layers = []

        for event in event_list:
            scope_data = self._parse_scope_data(event)
            if not scope_data:
                continue
            scope_names, start_time, dur_time = scope_data
            end_time = start_time + dur_time
            if layers and start_time < layers[0].ts + layers[0].dur:
                continue  # Skip parallel operators, keep only the first one

            merge = True  # Flag to control merging of upper layers
            for layer_depth, layer_name in enumerate(scope_names):
                if layer_depth >= len(layers):
                    layers.append(ScopeLayerCompleteEvent(
                        {"name": layer_name, "tid": layer_depth, "ts": start_time, "dur": dur_time}))
                    continue
                if merge and layers[layer_depth].name == layer_name:
                    layers[layer_depth].dur = end_time - layers[layer_depth].ts
                else:
                    pool.add_event(layers[layer_depth])
                    layers[layer_depth] = ScopeLayerCompleteEvent(
                        {"name": layer_name, "tid": layer_depth, "ts": start_time, "dur": dur_time})
                    merge = False

        # Add remaining layers to pool
        for layer in layers:
            pool.add_event(layer)

    @staticmethod
    def _parse_scope_data(event: CompleteEvent) -> Optional[Tuple[List[str], Decimal, Decimal]]:
        """Parse scope names and timing from event.

        Args:
            event (CompleteEvent): Event to parse.

        Returns:
            Optional[Tuple[List[str], Decimal, Decimal]]: Scope names, start time and duration.
        """
        if hasattr(event, 'parent') and event.parent:
            event_scope_name = event.name.split("/")[:-1]
            parent_scope_name = event.parent.name.split("::")[-1].split("/")[:-1]
            if event_scope_name and parent_scope_name:
                scope_name = (
                    parent_scope_name
                    if len(parent_scope_name) > len(event_scope_name)
                    else event_scope_name
                )
            else:
                scope_name = event_scope_name or parent_scope_name
        else:
            scope_name = event.name.split("/")[:-1]

        if scope_name:
            return scope_name, event.ts, event.dur
        return None

    @staticmethod
    def _create_meta_event(pool: TimelineEventPool) -> None:
        """Create meta events for scope layer."""
        process_meta_events = [
            (EventConstant.PROCESS_NAME, {"name": TimelineLayerName.SCOPER_LAYER.value}),
            (EventConstant.PROCESS_SORT, {"sort_index": EventConstant.SCOPE_LAYER_SORT_IDX})
        ]

        for name, args in process_meta_events:
            pool.add_event(ScopeLayerMetaEvent({"name": name, "tid": 0, "args": args}))

        for tid in pool.get_all_tids():
            pool.add_event(ScopeLayerMetaEvent({"name": EventConstant.THREAD_NAME, "tid": tid,
                                                "args": {"name": f"Thread {tid}"}}))
            pool.add_event(ScopeLayerMetaEvent({"name": EventConstant.THREAD_SORT, "tid": tid,
                                                "args": {"sort_index": tid}}))


def is_scope_data(event: BaseEvent) -> bool:
    """Check if event is scope data."""
    if isinstance(event, (CpuOpCompleteEvent, MsprofCompleteEvent)):
        scope_full_name = event.name
        return scope_full_name and scope_full_name.startswith(EventConstant.TOP_SCOPE_NAMES)

    if isinstance(event, FwkCompleteEvent):
        scope_full_name = event.name.split("::")[-1]
        return scope_full_name and scope_full_name.startswith(EventConstant.TOP_SCOPE_NAMES)

    return False
