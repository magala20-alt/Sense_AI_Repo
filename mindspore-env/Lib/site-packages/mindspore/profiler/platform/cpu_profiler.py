# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
"""CPU platform profiler."""
import mindspore._c_expression as c_expression

from mindspore.profiler.common.registry import PROFILERS
from mindspore.profiler.common.constant import DeviceTarget, ProfilerActivity, AnalysisMode
from mindspore.profiler.common.util import print_msg_with_pid
from mindspore.profiler.common.profiler_context import ProfilerContext
from mindspore.profiler.common.profiler_path_manager import ProfilerPathManager
from mindspore.profiler.common.process_pool import MultiProcessPool
from mindspore.profiler.platform.base_profiler import BaseProfiler
from mindspore.profiler.analysis.time_converter import TimeConverter
from mindspore.profiler.analysis.task_manager import TaskManager
from mindspore.profiler.analysis.parser.ms_framework_parser import FrameworkParser
from mindspore.profiler.analysis.parser.framework_cann_relation_parser import FrameworkCannRelationParser
from mindspore.profiler.analysis.viewer.ms_dataset_viewer import MsDatasetViewer
from mindspore.profiler.analysis.viewer.ascend_timeline_viewer import AscendTimelineViewer
from mindspore.profiler.common.log import ProfilerLogger


@PROFILERS.register_module(DeviceTarget.CPU.value)
class CpuProfiler(BaseProfiler):
    """
    CPU platform profiler
    """

    def __init__(self) -> None:
        super().__init__()
        self._prof_ctx = ProfilerContext()
        self._profiler = c_expression.Profiler.get_instance(DeviceTarget.CPU.value)
        self._prof_path_mgr = ProfilerPathManager()
        self._prof_mgr = c_expression.ProfilerManager.get_instance()
        ProfilerLogger.init(self._prof_ctx.ascend_ms_dir)
        self._logger = ProfilerLogger.get_instance()

    def start(self) -> None:
        """Start profiling."""
        self._logger.info("CpuProfiler start.")
        self._profiler.init(self._prof_ctx.framework_path)
        self._logger.info("CpuProfiler framework_path: %s", self._prof_ctx.framework_path)
        self._profiler.step_profiling_enable(True)

        if ProfilerActivity.CPU in self._prof_ctx.activities:
            self._profiler.enable_op_time()

        if self._prof_ctx.profile_memory:
            self._profiler.enable_profile_memory()

    def stop(self) -> None:
        """Stop profiling."""
        self._logger.info("CpuProfiler stop.")
        self._profiler.stop()

    def analyse(self, **kwargs) -> None:
        """Analyse profiling data."""
        if ProfilerContext().device_target_set != {DeviceTarget.CPU.value}:
            return
        self._logger.info("CpuProfiler analyse.")
        CPUProfilerAnalysis.online_analyse(**kwargs)

    def finalize(self) -> None:
        """Finalize profiling data."""
        self._logger.info("CpuProfiler finalize.")


class CPUProfilerAnalysis:
    """
    CPU profiler analysis interface
    """

    @classmethod
    def online_analyse(cls, async_mode: bool = False):
        """
        Online analysis for CPU
        """
        cls._pre_analyse_online()
        if async_mode:
            ProfilerContext().mode = AnalysisMode.ASYNC_MODE.value
            MultiProcessPool().add_async_job(cls._run_tasks, **ProfilerContext().to_dict())
        else:
            ProfilerContext().mode = AnalysisMode.SYNC_MODE.value
            cls._run_tasks(**ProfilerContext().to_dict())

    @classmethod
    def _pre_analyse_online(cls):
        """
        Pre-process for online analysis
        """
        ProfilerPathManager().create_output_path()
        TimeConverter.init_parameters(freq=100.0, cntvct=0, localtime_diff=0)

    @classmethod
    def _run_tasks(cls, **kwargs) -> None:
        """
        Run tasks for online analysis
        """
        ascend_ms_dir = kwargs.get("ascend_ms_dir", "")
        print_msg_with_pid(f"Start parsing profiling data: {ascend_ms_dir}")
        task_mgr = cls._construct_task_mgr(**kwargs)
        task_mgr.run()

    @classmethod
    def _construct_task_mgr(cls, **kwargs) -> TaskManager:
        """
        Construct task manager based on activities and parameters
        """
        task_mgr = TaskManager()

        task_mgr.create_flow(
            FrameworkParser(**kwargs)
            .register_post_hook(MsDatasetViewer(**kwargs).save),
            FrameworkCannRelationParser()
            .register_post_hook(AscendTimelineViewer(**kwargs).save),
            flow_name="cpu_flow", show_process=True
        )

        return task_mgr
