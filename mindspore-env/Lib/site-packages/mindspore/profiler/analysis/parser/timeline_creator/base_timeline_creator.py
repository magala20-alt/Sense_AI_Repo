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
"""Base class for timeline event creators."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from mindspore.profiler.analysis.parser.timeline_event.timeline_event_pool import TimelineEventPool


class BaseTimelineCreator(ABC):
    """Base class for creating timeline event pools."""

    def __init__(self):
        self.event_pools: Dict[int, TimelineEventPool] = {}

    @abstractmethod
    def create(self, data: Any) -> None:
        """Create timeline event pools from input data."""

    def get_chrome_trace_data(self) -> List[Dict]:
        """Get all events in chrome trace format."""
        if not self.event_pools:
            return []
        chrome_trace_data = []
        for pool in self.event_pools.values():
            if pool:
                chrome_trace_data.extend(pool.get_all_events_with_trace_format())
        return chrome_trace_data

    def get_event_pools(self) -> Dict[int, TimelineEventPool]:
        """Get all timeline event pools."""
        return self.event_pools
