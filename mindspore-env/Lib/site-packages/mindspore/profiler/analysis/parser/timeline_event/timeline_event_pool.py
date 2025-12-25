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
# ============================================================================"""
"""Timeline event pool for managing and categorizing events."""
from typing import Dict, List, Set
from collections import defaultdict

from mindspore.profiler.common.constant import EventConstant
from mindspore.profiler.analysis.parser.timeline_event.base_event import (
    BaseEvent,
    MetaEvent,
)


class TimelineEventPool:
    """A container class that manages and categorizes events within a specific timeline lane (process).

    This class is responsible for:
    1. Storing and categorizing different types of events (complete, instant, counter, meta) by thread ID
    2. Managing timeline connection events within the same process
    3. Providing event indexing capabilities by thread ID or thread name
    4. Maintaining thread name mappings for the process

    Note:
        This class only handles runtime events, offline events are managed separately.
    """

    def __init__(self, pid):
        # process ID
        self.pid = pid
        self.name = ""
        # Event storage by phase
        self.complete_event: Dict[int, List[BaseEvent]] = defaultdict(list)
        self.instance_event: Dict[int, List[BaseEvent]] = defaultdict(list)
        self.counter_event: Dict[int, List[BaseEvent]] = defaultdict(list)
        self.meta_event: List[BaseEvent] = []
        # Store start/end events for timeline connections
        self.start_to_end_events_pairs: Dict[str, Dict[str, List[BaseEvent]]] = {}
        # Thread mappings
        self.tid_to_name: Dict[int, str] = {}
        self.name_to_tid: Dict[str, int] = {}

    def add_event(self, event: BaseEvent) -> None:
        """Add event to timeline based on its phase type."""
        if event.ph == EventConstant.COMPLETE_EVENT:
            self.complete_event[event.tid].append(event)
        elif event.ph == EventConstant.INSTANT_EVENT:
            self.instance_event[event.tid].append(event)
        elif event.ph == EventConstant.COUNTER_EVENT:
            self.counter_event[event.tid].append(event)
        elif event.ph == EventConstant.META_EVENT:
            self.meta_event.append(event)
            self._handle_meta_event(event)

    def _handle_meta_event(self, event: MetaEvent) -> None:
        """Update process and thread mappings from meta event."""
        if event.name == EventConstant.PROCESS_NAME:
            self.name = event.args.get("name", "")
        elif event.name == EventConstant.THREAD_NAME:
            tid = event.tid
            thread_name = event.args.get("name", "")
            if tid is not None and thread_name:
                self.tid_to_name[tid] = thread_name
                self.name_to_tid[thread_name] = tid

    def add_start_event(self, flow_key: str, event: BaseEvent) -> None:
        """Add start event for timeline connection."""
        if flow_key not in self.start_to_end_events_pairs:
            self.start_to_end_events_pairs[flow_key] = {"start": [], "end": []}
        self.start_to_end_events_pairs[flow_key]["start"].append(event)

    def add_end_event(self, flow_key: str, event: BaseEvent) -> None:
        """Add end event for timeline connection."""
        if flow_key not in self.start_to_end_events_pairs:
            self.start_to_end_events_pairs[flow_key] = {"start": [], "end": []}
        self.start_to_end_events_pairs[flow_key]["end"].append(event)

    @staticmethod
    def _get_events(event_dict: dict) -> List[BaseEvent]:
        """Helper function to get events from a dictionary."""
        events = []
        for event_list in event_dict.values():
            events.extend(event_list)
        return events

    def get_complete_events(self) -> List[BaseEvent]:
        """Get all complete events."""
        return self._get_events(self.complete_event)

    def get_instant_events(self) -> List[BaseEvent]:
        """Get all instant events."""
        return self._get_events(self.instance_event)

    def get_counter_events(self) -> List[BaseEvent]:
        """Get all counter events."""
        return self._get_events(self.counter_event)

    def get_all_events(self) -> List[BaseEvent]:
        """Get all events in order: meta events first, followed by complete, instant and counter events."""
        all_events = []
        all_events.extend(self.meta_event)
        all_events.extend(self.get_complete_events())
        all_events.extend(self.get_instant_events())
        all_events.extend(self.get_counter_events())
        return all_events

    def get_start_to_end_flow_pairs(self) -> Dict[str, Dict[str, List[BaseEvent]]]:
        """Get all start/end events for timeline connections."""
        return self.start_to_end_events_pairs

    def get_events_by_tid(self, tid: int) -> List[BaseEvent]:
        """Get all events for specified thread ID."""
        events = []
        events.extend(self.complete_event.get(tid, []))
        events.extend(self.instance_event.get(tid, []))
        events.extend(self.counter_event.get(tid, []))
        return events

    def get_events_by_name(self, name: str) -> List[BaseEvent]:
        """Get all events for specified thread name."""
        tid = self.name_to_tid.get(name)
        if tid is None:
            return []
        return self.get_events_by_tid(tid)

    def get_all_tids(self) -> Set[int]:
        """Get set of all thread IDs."""
        tids = set()
        tids.update(self.complete_event.keys())
        tids.update(self.instance_event.keys())
        tids.update(self.counter_event.keys())
        return tids

    def get_all_events_with_trace_format(self) -> List[Dict]:
        """Convert and return all events in Chrome trace format."""
        return [event.to_trace_format() for event in self.get_all_events()]
