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
"""Base class for all event types."""
from decimal import Decimal
from typing import Dict
from abc import ABC, abstractmethod

from mindspore.profiler.common.constant import EventConstant


class BaseEvent(ABC):
    """Base class for all event types."""

    def __init__(self, data: Dict):
        """Initialize base event with data dictionary."""
        if not isinstance(data, dict):
            raise TypeError("Input data must be dict.")
        self._origin_data = data

    @property
    def ph(self) -> str:
        """Get event phase."""
        raise NotImplementedError

    @property
    def origin_data(self) -> Dict:
        """Get original event data."""
        return self._origin_data

    @property
    def pid(self) -> int:
        """Get process ID."""
        return int(self._origin_data.get("pid", 0))

    @property
    def tid(self) -> int:
        """Get thread ID."""
        return int(self._origin_data.get("tid", 0))

    @property
    def name(self) -> str:
        """Get event name."""
        return self._origin_data.get("name", "")

    @abstractmethod
    def to_trace_format(self) -> dict:
        """Convert event to Chrome trace format."""
        raise NotImplementedError


class MetaEvent(BaseEvent):
    """Metadata event class."""

    @property
    def ph(self) -> str:
        """Get event phase (M for metadata)."""
        return EventConstant.META_EVENT

    @property
    def args(self) -> dict:
        """Get event arguments."""
        return self._origin_data.get("args", {})

    def to_trace_format(self) -> dict:
        """Convert metadata event to Chrome trace format."""
        return {
            "ph": self.ph,
            "name": self.name,
            "pid": self.pid,
            "tid": self.tid,
            "args": self.args
        }


class CompleteEvent(BaseEvent):
    """Complete event class."""

    @property
    def ph(self) -> str:
        """Get event phase (X for complete)."""
        return EventConstant.COMPLETE_EVENT

    @property
    def ts(self) -> Decimal:
        """Get event start timestamp."""
        return Decimal(self._origin_data.get("ts", 0))

    @property
    def dur(self) -> Decimal:
        """Get event duration."""
        return Decimal(self._origin_data.get("dur", 0))

    @property
    def cat(self) -> str:
        """Get event category."""
        return str(self._origin_data.get("cat", ""))

    @property
    def args(self) -> dict:
        """Get event arguments."""
        return self._origin_data.get("args", {})

    @property
    def unique_id(self) -> str:
        """Get unique id"""
        return f"{self.pid}-{self.tid}-{self.ts}"

    def to_trace_format(self) -> dict:
        """Convert complete event to Chrome trace format."""
        return {
            "ph": self.ph,
            "name": self.name,
            "pid": self.pid,
            "tid": self.tid,
            "ts": str(self.ts),
            "dur": str(self.dur),
            "cat": self.cat,
            "args": self.args
        }


class InstantEvent(BaseEvent):
    """Instant event class."""

    @property
    def ph(self) -> str:
        """Get event phase (i for instant)."""
        return EventConstant.INSTANT_EVENT

    @property
    def ts(self) -> Decimal:
        """Get event timestamp."""
        return Decimal(self._origin_data.get("ts", 0))

    @property
    def args(self) -> dict:
        """Get event arguments."""
        return self._origin_data.get("args", {})

    def to_trace_format(self) -> dict:
        """Convert instant event to Chrome trace format."""
        return {
            "name": self.name,
            "ph": self.ph,
            "ts": str(self.ts),
            "pid": self.pid,
            "tid": self.tid,
            "args": self.args
        }


class CounterEvent(BaseEvent):
    """Counter event class."""

    @property
    def ph(self) -> str:
        """Get event phase (C for counter)."""
        return EventConstant.COUNTER_EVENT

    @property
    def ts(self) -> Decimal:
        """Get event timestamp."""
        return Decimal(self._origin_data.get("ts", 0))

    @property
    def args(self) -> dict:
        """Get event arguments."""
        return self._origin_data.get("args", {})

    def to_trace_format(self) -> dict:
        """Convert counter event to Chrome trace format."""
        return {
            "name": self.name,
            "ph": self.ph,
            "ts": str(self.ts),
            "pid": self.pid,
            "tid": self.tid,
            "args": self.args
        }


class FlowEvent(BaseEvent):
    """Flow event class."""

    @property
    def ph(self) -> str:
        """Get event phase (s/t/f for flow start/step/end)."""
        raise NotImplementedError

    @property
    def flow_id(self) -> str:
        """Get flow identifier."""
        return self._origin_data.get("id", "")

    @property
    def ts(self) -> Decimal:
        """Get event timestamp."""
        return Decimal(self._origin_data.get("ts", 0))

    @property
    def cat(self) -> str:
        """Get event category."""
        return str(self._origin_data.get("cat", ""))

    @property
    def unique_id(self) -> str:
        """Get unique id"""
        return f"{self.pid}-{self.tid}-{self.ts}"

    def to_trace_format(self) -> dict:
        """Convert flow event to Chrome trace format."""
        return {
            "name": self.name,
            "bp": "e",
            "ph": self.ph,
            "ts": str(self.ts),
            "pid": self.pid,
            "tid": self.tid,
            "id": self.flow_id,
            "cat": self.cat
        }
