# Copyright 2022-2024 Huawei Technologies Co., Ltd
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
"""Profiler Info"""
import os
from typing import Dict, Any
from configparser import ConfigParser, NoSectionError, NoOptionError

from mindspore import log as logger
from mindspore.version import __version__ as ms_version
from mindspore.profiler.common.util import get_cann_version
from mindspore.profiler.common.singleton import Singleton
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.path_manager import PathManager
from mindspore.profiler.common.msprof_cmd_tool import MsprofCmdTool


@Singleton
class ProfilerInfo:
    """
    Profiler Info
    This class is used to record profiler information.

    Example:
    {
        "profiler_parameters": {
        },
        "ms_profiler_info": {
            "start_collect_syscnt": xxx,
            "start_collect_clock_time": xxx,
        },
        "cann_profiler_info": {
        },
        "ms_version": "x.x.x",
        "cann_version": x.x.x",
    }
    """
    HOST_START_LOG = "host_start.log"
    START_INFO = "start_info"
    MSPROF_INFO_SCRIPT_PATH = os.path.join(
        "tools",
        "profiler",
        "profiler_tool",
        "analysis",
        "interface",
        "get_msprof_info.py",
    )

    PROFILER_INFO_FILE = "profiler_info_{}.json"
    PROFILER_PARAMETERS = "profiler_parameters"
    MS_PROFILER_INFO = "ms_profiler_info"
    CANN_PROFILER_INFO = "cann_profiler_info"
    ANALYSIS_COST_TIME = "analysis_cost_time"
    MS_VERSION = "ms_version"
    CANN_VERSION = "cann_version"
    JIT_LEVEL = "jit_level"

    US_TO_NS = 1000

    def __init__(self):
        self._profiler_info = {
            self.PROFILER_PARAMETERS: {},
            self.MS_PROFILER_INFO: {},
            self.CANN_PROFILER_INFO: {},
            self.ANALYSIS_COST_TIME: {},
            self.MS_VERSION: ms_version,
            self.CANN_VERSION: get_cann_version(),
            self.JIT_LEVEL: "",
        }
        # time params
        self._freq = 100.0
        self._cntvct = 0
        self._clock_monotonic_raw = 0 # from host start log
        self._collection_time_begin = 0
        self._clock_monotonic_raw_info = 0 # from start info file
        self._localtime_diff = 0

    def load_info(self, info_file_path: str) -> None:
        """"
        Load profiler info from profiler_info_*.json path.
        """
        self._profiler_info = FileManager.read_json_file(info_file_path)

    def load_time_parameters(self, msprof_profile_path: str, msprof_profile_host_path: str) -> None:
        """
        Load time parameters from msprof profile and host start log.
        This method should be called before TimeConverter.init_parameters.
        """
        msprof_info = MsprofCmdTool(msprof_profile_path).get_msprof_info()
        if not msprof_profile_path or not msprof_profile_host_path:
            raise ValueError(
                "msprof_profile_path and msprof_profile_host_path must be provided"
            )
        self._read_host_start_log(msprof_profile_host_path)
        self._read_start_info(msprof_profile_host_path)
        self._get_freq_from_msprof(msprof_info)

    @property
    def time_parameters(self) -> Dict[str, Any]:
        """
        Get time parameters for TimeConverter.
        """
        return {
            "freq": self._freq,
            "cntvct": self._cntvct,
            "localtime_diff": self._localtime_diff
        }

    @property
    def profiler_parameters(self) -> Dict[str, str]:
        """
        Get profiler parameters.
        """
        return self._profiler_info[self.PROFILER_PARAMETERS]

    @profiler_parameters.setter
    def profiler_parameters(self, value: dict):
        """
        Set profiler parameters.
        """
        self._profiler_info[self.PROFILER_PARAMETERS] = value

    @property
    def jit_level(self) -> str:
        """
        Get jit level.
        """
        return self._profiler_info[self.JIT_LEVEL]

    @jit_level.setter
    def jit_level(self, value: str):
        """
        Set jit level.
        """
        self._profiler_info[self.JIT_LEVEL] = value

    @property
    def ms_profiler_info(self) -> Dict[str, str]:
        """
        Get ms profiler info.
        """
        return self._profiler_info[self.MS_PROFILER_INFO]

    @ms_profiler_info.setter
    def ms_profiler_info(self, value: dict):
        """
        Set ms profiler info.
        """
        self._profiler_info[self.MS_PROFILER_INFO] = value

    def save(self, output_path: str, rank_id: int):
        """
        Save profiler info to json file.
        """
        if not output_path:
            logger.warning("Output path is empty, please check the output path.")
            return

        FileManager.create_json_file(
            output_file_path=os.path.join(
                output_path, self.PROFILER_INFO_FILE.format(rank_id)
            ),
            json_data=self._profiler_info,
            indent=4,
        )

    def _read_host_start_log(self, host_path: str) -> None:
        """
        Read host_start.log and get clock_monotonic_raw and cntvct

        host_start.log format:
        [Host]
        clock_monotonic_raw = 1234567890
        cntvct = 1234567890
        cntvct_diff = 1234567890
        """
        start_log_path = os.path.join(host_path, self.HOST_START_LOG)
        PathManager.check_input_file_path(start_log_path)

        cfg = ConfigParser()
        try:
            cfg.read(start_log_path)
        except Exception as err: # pylint: disable=W0703
            raise ValueError(f"Error parsing host start log file: {err}") from err

        try:
            self._clock_monotonic_raw = int(cfg.get("Host", "clock_monotonic_raw"))
            self._cntvct = int(cfg.get("Host", "cntvct"))
        except NoSectionError as err:
            raise ValueError("'Host' section not found in host start log file") from err
        except NoOptionError as err:
            raise ValueError(f"Required option not found in host start log file: {err}") from err
        except ValueError as err:
            raise ValueError(f"Invalid data in host start log file: {err}") from err

        if self._clock_monotonic_raw <= 0 or self._cntvct <= 0:
            raise ValueError(
                "Invalid clock_monotonic_raw or cntvct value must be positive"
            )

    def _read_start_info(self, host_path: str) -> None:
        """
        Read start_info file and calculate localtime_diff.

        Raises:
            FileNotFoundError: If the start_info file is not found.
            ValueError: If required data is missing or invalid in the file.
            JSONDecodeError: If there's an error decoding the JSON file.
        """
        start_info_path = os.path.join(host_path, self.START_INFO)
        info_data = FileManager.read_json_file(start_info_path)

        try:
            self._collection_time_begin = int(
                info_data.get("collectionTimeBegin", 0)
            )  # us
            self._clock_monotonic_raw_info = int(info_data.get("clockMonotonicRaw", 0))
        except ValueError as err:
            raise ValueError(f"Invalid data in start info file: {err}") from err

        if self._collection_time_begin <= 0 or self._clock_monotonic_raw_info <= 0:
            raise ValueError(
                "Invalid collectionTimeBegin or clockMonotonicRaw value must be positive"
            )

        self._localtime_diff = self._clock_monotonic_raw + (
            self._collection_time_begin * self.US_TO_NS - self._clock_monotonic_raw_info
        )

    def _get_freq_from_msprof(self, msprof_info: str) -> None:
        """
        Get frequency from get_msprof_info.py script

        script return json information:
        {
            "data": {
                "host_info": {
                    "cpu_info": [{"Frequency": "100.0000"}]
                }
            }
        }
        """

        if not isinstance(msprof_info, dict):
            raise RuntimeError("msprof_info must be a dictionary")

        data = msprof_info.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("msprof_info['data'] must be a dictionary")

        host_info = data.get("host_info")
        if not isinstance(host_info, dict):
            raise RuntimeError("msprof_info['data']['host_info'] must be a dictionary")

        cpu_info_list = host_info.get("cpu_info", [])
        if not isinstance(cpu_info_list, list) or not cpu_info_list:
            raise RuntimeError("cpu_info must be a non-empty list")

        cpu_info = cpu_info_list[0]
        if not isinstance(cpu_info, dict):
            raise RuntimeError("cpu_info[0] must be a dictionary")

        freq_str = cpu_info.get("Frequency", self._freq)
        if not freq_str:
            logger.warning("Frequency is empty, use default frequency: %s", str(self._freq))
            return

        try:
            freq = float(freq_str)
        except ValueError as e:
            raise ValueError(
                "Convert frequency to float failed, please check msprof information"
            ) from e

        if freq <= 0.0:
            raise ValueError("Frequency is invalid, please check msprof info")
        self._freq = freq
