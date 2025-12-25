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
"""Profiler output path"""
import os
import glob
import re
from typing import Any, Dict, Optional
from mindspore import log as logger
from mindspore.profiler.common.path_manager import PathManager


class ProfilerOutputPath:
    """
    Profiler output path structure:

    └── output_path
        └── {}_ascend_ms  # rank 0
                └── ASCEND_PROFILER_OUTPUT
                └── FRAMEWORK
                └── PROF_{}
                        └── host
                        └── device_{}
                        └── mindstudio_profiler_log
                        └── mindstudio_profiler_output
        └── {}_ascend_ms  # rank 1
                └── ASCEND_PROFILER_OUTPUT
                └── FRAMEWORK
                └── PROF_{}
                        └── host
                        └── device_{}
                        └── mindstudio_profiler_log
                        └── mindstudio_profiler_output
    """

    _ASCEND_PROFILER_OUTPUT = "ASCEND_PROFILER_OUTPUT"
    _FRAMEWORK = "FRAMEWORK"
    _MINISTUDIO_PROFILER_HOST = "host"
    _MINISTUDIO_PROFILER_DEVICE = "device_{}"
    _MINISTUDIO_PROFILER_LOG = "mindstudio_profiler_log"
    _MINISTUDIO_PROFILER_OUTPUT = "mindstudio_profiler_output"
    _MINISTUDIO_ANALYZE_OUTPUT = "analyze"

    def __init__(self, rank_id: int):

        self._rank_id = rank_id
        self._output_path: Optional[str] = None
        self._ascend_ms_dir: Optional[str] = None
        self._ascend_profiler_output_path: Optional[str] = None
        self._framework_path: Optional[str] = None

        # PROF_{} and its subdirectories
        self._msprof_profile_path: Optional[str] = None
        self._msprof_profile_host_path: Optional[str] = None
        self._msprof_profile_device_path: Optional[str] = None
        self._msprof_profile_log_path: Optional[str] = None
        self._msprof_profile_output_path: Optional[str] = None
        self._msprof_analyze_output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the profiler parameters to a dictionary.
        """
        return {
            "output_path": self._output_path,
            "ascend_ms_dir": self._ascend_ms_dir,
            "ascend_profiler_output_path": self._ascend_profiler_output_path,
            "framework_path": self._framework_path,
            "msprof_profile_path": self._msprof_profile_path,
            "msprof_profile_host_path": self._msprof_profile_host_path,
            "msprof_profile_device_path": self._msprof_profile_device_path,
            "msprof_profile_log_path": self._msprof_profile_log_path,
            "msprof_profile_output_path": self._msprof_profile_output_path,
            "msprof_analyze_output_path": self._msprof_analyze_output_path,
        }

    @property
    def output_path(self) -> str:
        """
        Get the output path, which is the parent directory of all profiler output paths.
        Set by Profiler output_path parameter.

        Returns:
            str: The output path.

        Raises:
            ValueError: If output_path has not been set.
        """
        if self._output_path is None:
            raise ValueError("output_path has not been set")
        return self._output_path

    @property
    def ascend_ms_dir(self) -> str:
        """
        Get the Ascend MS directory, which is each rank's output path.

        Returns:
            str: The Ascend MS directory, eg. output_path/xxxx_ascend_ms

        Raises:
            ValueError: If ascend_ms_dir has not been set.
        """
        if self._ascend_ms_dir is None:
            raise ValueError("ascend_ms_dir has not been set")
        return self._ascend_ms_dir

    @property
    def ascend_profiler_output_path(self) -> str:
        """
        Get the MindSpore Profiler official deliverables output path.

        Returns:
            str: The MindSpore Profiler official deliverables output path.
            eg. ascend_ms_dir/ASCEND_PROFILER_OUTPUT

        Raises:
            ValueError: If ascend_profiler_output_path has not been set.
        """
        if self._ascend_profiler_output_path is None:
            raise ValueError("ascend_profiler_output_path has not been set")
        return self._ascend_profiler_output_path

    @property
    def framework_path(self) -> str:
        """
        Get the framework path, which is the intermediate directory of framework profiling data.

        Returns:
            str: The framework path. eg. ascend_ms_dir/FRAMEWORK

        Raises:
            ValueError: If framework_path has not been set.
        """
        if self._framework_path is None:
            raise ValueError("framework_path has not been set")
        return self._framework_path

    @property
    def msprof_profile_path(self) -> str:
        """
        Get the PROF_XXX path, which is the directory of msprof profiling data.

        Returns:
            str: The MSProf profile path. eg. ascend_ms_dir/PROF_XXX

        Raises:
            ValueError: If msprof_profile_path has not been set.
        """
        if self._msprof_profile_path is None:
            raise ValueError("msprof_profile_path has not been set")
        return self._msprof_profile_path

    @property
    def msprof_profile_host_path(self) -> str:
        """
        Get the msprof host path.

        Returns:
            str: The msprof host path. eg. ascend_ms_dir/PROF_XXX/host

        Raises:
            ValueError: If msprof_profile_host_path has not been set.
        """
        if self._msprof_profile_host_path is None:
            raise ValueError("msprof_profile_host_path has not been set")
        return self._msprof_profile_host_path

    @property
    def msprof_profile_device_path(self) -> str:
        """
        Get the msprof device path.

        Returns:
            str: The msprof device path. eg. ascend_ms_dir/PROF_XXX/device_X

        Raises:
            ValueError: If msprof_profile_device_path has not been set.
        """
        if self._msprof_profile_device_path is None:
            raise ValueError("msprof_profile_device_path has not been set")
        return self._msprof_profile_device_path

    @property
    def msprof_profile_log_path(self) -> str:
        """
        Get the msprof log path.

        Returns:
            str: The msprof log path. eg. ascend_ms_dir/PROF_XXX/mindstudio_profiler_log

        Raises:
            ValueError: If msprof_profile_log_path has not been set.
        """
        if self._msprof_profile_log_path is None:
            raise ValueError("msprof_profile_log_path has not been set")
        return self._msprof_profile_log_path

    @property
    def msprof_profile_output_path(self) -> str:
        """
        Get the msprof official deliverables output path.

        Returns:
            str: The msprof official deliverables output path.
            eg. ascend_ms_dir/PROF_XXX/mindstudio_profiler_output

        Raises:
            ValueError: If msprof_profile_output_path has not been set.
        """
        if self._msprof_profile_output_path is None:
            raise ValueError("msprof_profile_output_path has not been set")
        return self._msprof_profile_output_path

    @property
    def msprof_analyze_output_path(self) -> str:
        """
        Get the msprof analyze output path.

        Returns:
            str: The msprof analyze output path.
            eg. ascend_ms_dir/PROF_XXX/analyze

        Raises:
            ValueError: If msprof_analyze_output_path has not been set.
        """
        if self._msprof_analyze_output_path is None:
            raise ValueError("msprof_analyze_output_path has not been set")
        return self._msprof_analyze_output_path

    @output_path.setter
    def output_path(self, value: str):
        """Set the output path."""
        real_path = PathManager.get_real_path(value)
        PathManager.check_input_directory_path(real_path)
        self._output_path = real_path

    @ascend_ms_dir.setter
    def ascend_ms_dir(self, value: str):
        """Set the xxx_ascend_ms directory."""
        if self._output_path is None:
            raise ValueError("output_path has not been set")

        self._ascend_ms_dir = os.path.join(self._output_path, value)
        self._ascend_profiler_output_path = os.path.join(
            self._ascend_ms_dir, ProfilerOutputPath._ASCEND_PROFILER_OUTPUT
        )
        self._framework_path = os.path.join(
            self._ascend_ms_dir, ProfilerOutputPath._FRAMEWORK
        )

    @msprof_profile_path.setter
    def msprof_profile_path(self, value: str):
        """Set the PROF_XXX path."""
        if self._ascend_ms_dir is None:
            raise ValueError("ascend_ms_dir has not been set")
        self._msprof_profile_path = value
        self._msprof_profile_host_path = os.path.join(
            self._msprof_profile_path, ProfilerOutputPath._MINISTUDIO_PROFILER_HOST
        )

        device_pattern = os.path.join(self._msprof_profile_path, "device_*")
        device_dirs = glob.glob(device_pattern)
        valid_device_dirs = [d for d in device_dirs if re.match(r'^device_\d+$', os.path.basename(d))]

        if valid_device_dirs:
            device_dir = os.path.basename(valid_device_dirs[0])
            device_id = device_dir.replace("device_", "")
            self._msprof_profile_device_path = os.path.join(
                self._msprof_profile_path,
                ProfilerOutputPath._MINISTUDIO_PROFILER_DEVICE.format(device_id)
            )
        else:
            logger.error(f"No device_* directory found in {self._msprof_profile_path}, using device_0 as default")
            self._msprof_profile_device_path = os.path.join(
                self._msprof_profile_path,
                ProfilerOutputPath._MINISTUDIO_PROFILER_DEVICE.format("0")
            )

        self._msprof_profile_log_path = os.path.join(
            self._msprof_profile_path, ProfilerOutputPath._MINISTUDIO_PROFILER_LOG
        )
        self._msprof_profile_output_path = os.path.join(
            self._msprof_profile_path, ProfilerOutputPath._MINISTUDIO_PROFILER_OUTPUT
        )
        self._msprof_analyze_output_path = os.path.join(
            self._msprof_profile_path, ProfilerOutputPath._MINISTUDIO_ANALYZE_OUTPUT
        )
