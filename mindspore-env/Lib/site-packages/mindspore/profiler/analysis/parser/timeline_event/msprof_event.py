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
"""Msprof event classes for MindSpore profiling data."""

from typing import Dict, List, Optional
from mindspore.profiler.analysis.parser.timeline_event.base_event import (
    BaseEvent,
    MetaEvent,
    InstantEvent,
    CompleteEvent,
    CounterEvent
)


class MsprofCompleteEvent(CompleteEvent):
    """Msprof Complete event class for representing complete operations with duration."""
    _MINDSPORE_OP_KEY = "mindspore_op"

    def __init__(self, data: Dict):
        """Initialize complete event with data and empty event references."""
        super().__init__(data)
        self._parent: Optional[BaseEvent] = None
        self._children: List[BaseEvent] = []
        self._step_id = None

    @property
    def parent(self) -> BaseEvent:
        """Get parent event reference."""
        return self._parent

    @parent.setter
    def parent(self, event: BaseEvent) -> None:
        """Set parent event and update mindspore_op argument."""
        self._parent = event
        self.args.update({self._MINDSPORE_OP_KEY: event.name})

    @property
    def children(self) -> List[BaseEvent]:
        """Get list of children event references."""
        return self._children

    @property
    def step_id(self) -> str:
        """Get parent event reference."""
        return self._step_id

    @step_id.setter
    def step_id(self, value):
        self._step_id = value


class MsprofInstantEvent(InstantEvent):
    """Msprof Instant event class for representing instantaneous operations."""


class MsprofMetaEvent(MetaEvent):
    """Msprof Meta event class for representing metadata information."""


class MsprofCounterEvent(CounterEvent):
    """Msprof Counter event class for representing counter-based metrics."""
