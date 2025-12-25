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
"""Flow event classes for representing event relationships in Chrome trace format."""

from mindspore.profiler.common.constant import EventConstant
from mindspore.profiler.analysis.parser.timeline_event.base_event import FlowEvent


class FlowStartEvent(FlowEvent):
    """Flow start event class for marking the beginning of a flow relationship."""

    @property
    def ph(self) -> str:
        """Get event phase ('s' for flow start)."""
        return EventConstant.START_FLOW


class FlowEndEvent(FlowEvent):
    """Flow end event class for marking the end of a flow relationship."""

    @property
    def ph(self) -> str:
        """Get event phase ('f' for flow end)."""
        return EventConstant.END_FLOW
