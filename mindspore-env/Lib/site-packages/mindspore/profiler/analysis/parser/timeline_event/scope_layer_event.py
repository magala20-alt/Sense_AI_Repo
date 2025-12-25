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
"""Scope event class for representing scope layer operations in profiling data."""

from decimal import Decimal

from mindspore.profiler.common.constant import EventConstant
from mindspore.profiler.analysis.parser.timeline_event.base_event import CompleteEvent, MetaEvent


class ScopeLayerCompleteEvent(CompleteEvent):
    """Scope layer complete event class for representing scope operations with duration."""

    def __init__(self, data: dict):
        """Initialize scope layer event with data and duration."""
        super().__init__(data)
        self._dur = Decimal(data.get('dur', 0))

    @property
    def dur(self) -> Decimal:
        """Get scope operation duration."""
        return self._dur

    @dur.setter
    def dur(self, value: Decimal):
        """Set scope operation duration."""
        self._dur = value

    @property
    def pid(self) -> int:
        """Get scope layer process ID."""
        return int(EventConstant.SCOPE_LAYER_PID)


class ScopeLayerMetaEvent(MetaEvent):
    """Scope layer meta event class for scope layer metadata."""

    @property
    def pid(self) -> int:
        """Get scope layer process ID."""
        return int(EventConstant.SCOPE_LAYER_PID)
