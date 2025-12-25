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
"""CPU event class."""
from decimal import Decimal

from mindspore.profiler.common.constant import TimeConstant, EventConstant
from mindspore.profiler.analysis.parser.timeline_event.base_event import CompleteEvent, MetaEvent


class CpuOpCompleteEvent(CompleteEvent):
    """CPU Complete(X) event class for representing CPU operations."""

    @property
    def ts(self) -> Decimal:
        """Get timestamp in microseconds, converting from nanoseconds."""
        return (Decimal(self._origin_data.get("ts", 0)) * Decimal(TimeConstant.NS_TO_US)).quantize(Decimal('0.000'))

    @property
    def dur(self) -> Decimal:
        """Get duration in microseconds, converting from milliseconds."""
        return (Decimal(self._origin_data.get("dur", 0)) * Decimal(TimeConstant.MS_TO_US)).quantize(Decimal('0.000'))

    @property
    def pid(self) -> int:
        """Get CPU operation process ID."""
        return int(EventConstant.CPU_OP_PID)


class CpuOpMetaEvent(MetaEvent):
    """CPU Meta(M) event class for CPU operation metadata."""

    @property
    def pid(self) -> int:
        """Get CPU operation process ID."""
        return int(EventConstant.CPU_OP_PID)
