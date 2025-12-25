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
"""Time converter"""
from decimal import Decimal


class TimeConverter:
    """
    Time converter class use to convert syscnt to timestamp

    The timeline is as follows:


                  boot                collect start       event
                    │                     │                │
     origin─────────┴─────────────────────┴────────────────┴─────────────►

           ────────────────────────────────
                    local_time_diff

                                          ──────────────────
                                        cntvct            syscnt


    boot: The point of events that occurred at the boot time of server
    collect start: The point of events that occurred at the profiler collect start time
    event: The point of events that occurred at the profiler record event time

    schematic:
    MindSpore profiler record event time is syscnt, we need to convert it to timestamp(us).
    First, we get the boot time(collectionTimeBegin and clockMonotonicRaw) and
    collect start time(cntvct and clock_monotonic_raw) from host_start.log,
    then we can calculate the local_time_diff = clock_monotonic_raw + (collectionTimeBegin - clockMonotonicRaw)
    Finally, we can get the profiler record absolute syscnt = syscnt - cntvct + local_time_diff, and convert
    it to timestamp(ns) by formula: timestamp(ns) = syscnt * 1000 / frequency, 1000 means convert ns to us.
    """
    # multiplier
    US_TO_NS = 1000
    NS_TO_US = Decimal("1e-3")
    DECIMAL_PRECISION = Decimal("0.000")
    # parameters from msprof
    _freq = 100.0
    _cntvct = 0
    _localtime_diff = 0
    _is_loaded = False

    @classmethod
    def convert_syscnt_to_timestamp_us(cls, syscnt: int) -> Decimal:
        """
        Convert syscnt to timestamp(us)
        Args:
            syscnt: syscnt
            time_fmt: time format
        Returns:
            timestamp(us)
        """
        if not cls._is_loaded:
            raise RuntimeError("init_parameters must be called first")

        timestamp_ns = Decimal(
            (syscnt - cls._cntvct) * cls.US_TO_NS / cls._freq
        ) + Decimal(cls._localtime_diff)

        timestamp_us = timestamp_ns * cls.NS_TO_US
        return timestamp_us.quantize(cls.DECIMAL_PRECISION)

    @classmethod
    def init_parameters(cls, freq: float, cntvct: int, localtime_diff: int):
        cls._freq = freq
        cls._cntvct = cntvct
        cls._localtime_diff = localtime_diff
        cls._is_loaded = True
