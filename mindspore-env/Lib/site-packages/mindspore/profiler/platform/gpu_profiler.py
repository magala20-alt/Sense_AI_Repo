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
"""GPU platform profiler."""
import mindspore._c_dataengine as cde
import mindspore._c_expression as c_expression
from mindspore.profiler.common.registry import PROFILERS
from mindspore.profiler.common.constant import DeviceTarget, ProfilerActivity

from mindspore.profiler.common.profiler_context import ProfilerContext
from mindspore.profiler.platform.base_profiler import BaseProfiler
from mindspore.profiler.common.log import ProfilerLogger


@PROFILERS.register_module(DeviceTarget.GPU.value)
class GpuProfiler(BaseProfiler):
    """
    GPU platform profiler
    """
    def __init__(self) -> None:
        super().__init__()
        self._prof_ctx = ProfilerContext()
        self._profiler = c_expression.Profiler.get_instance(DeviceTarget.GPU.value)
        ProfilerLogger.init(self._prof_ctx.ascend_ms_dir)
        self._logger = ProfilerLogger.get_instance()

        self._profiler.init(self._prof_ctx.output_path)
        self._profiler.sync_enable(self._prof_ctx.sync_enable)

        if self._prof_ctx.data_process:
            self._md_profiler = cde.GlobalContext.profiling_manager()
            self._md_profiler.init()

    def start(self) -> None:
        """Start profiling."""
        self._logger.info("GpuProfiler start.")

        if self._prof_ctx.data_process:
            self._profiler.data_process_enable(True)
            self._md_profiler.start()

        if ProfilerActivity.GPU in self._prof_ctx.activities:
            self._profiler.enable_op_time()

            if ProfilerActivity.CPU in self._prof_ctx.activities:
                self._profiler.step_profiling_enable(True)

    def stop(self) -> None:
        """Stop profiling."""
        self._logger.info("GpuProfiler stop.")
        self._profiler.stop()

        if self._prof_ctx.data_process:
            self._md_profiler.stop()
            self._md_profiler.save(self._prof_ctx.output_path)

    def analyse(self, **kwargs) -> None:
        """Analyse profiling data."""
        self._logger.info("GpuProfiler analyse.")

    def finalize(self) -> None:
        """Finalize profiling data."""
        self._logger.info("GpuProfiler finalize.")
