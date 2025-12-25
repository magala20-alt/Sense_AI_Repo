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
"""Framework event classes for timeline analysis."""
from enum import Enum
from decimal import Decimal
from typing import Dict, Optional, List

from mindspore import log as logger
from mindspore.profiler.common.constant import EventConstant, FileConstant
from mindspore.profiler.analysis.time_converter import TimeConverter
from mindspore.profiler.analysis.parser.timeline_event.base_event import (
    BaseEvent,
    CompleteEvent,
    MetaEvent,
    InstantEvent,
)


class ProfilerStage(Enum):
    """Profiler stage enumeration."""
    DEFAULT = "Default"
    PYTHON = "Python"
    CAPTURE = "Capture"
    RUN_GRAPH = "RunGraph"
    RUN_GRAD = "RunGrad"
    RUN_OP = "RunOp"
    ASNUMPY = "Asnumpy"
    COMPILE_GRAD_GRAPH = "CompileGradGraph"
    WAIT_PIPELINE = "WaitPipeline"
    SYNC_STREAM = "SyncStream"


class ProfilerModule(Enum):
    """Profiler module enumeration."""
    DEFAULT = "Default"
    GRAPH_EXECUTOR_PY = "GraphExecutorPy"
    RUNTIME_FRAMEWORK = "RuntimeFramework"
    PYNATIVE_FRAMEWORK = "PynativeFramework"
    KERNEL = "Kernel"
    PYTHON = "Python"
    CAPTURE = "Capture"
    OTHER = "Other"


class ProfilerEvent(Enum):
    """Profiler event enumeration."""
    DEFAULT = "Default"
    KERNEL_INFER = "KernelInfer"
    KERNEL_RESIZE = "KernelResize"
    KERNEL_INFER_AND_RESIZE = "KernelInferAndResize"
    KERNEL_LAUNCH = "KernelLaunch"
    KERNEL_LAUNCH_CALLBACK = "KernelLaunckCallback"
    KERNEL_UPDATE = "KernelUpdate"
    KERNEL_PREPARE_DATA = "KernelPrepareData"
    GRAPH_LAUNCH = "GraphLaunch"
    INPUT_PROCESS = "InputProcess"
    OUTPUT_PROCESS = "OutputProcess"
    WAIT_TASK_FINISH = "WaitTaskFinish"
    PRE_LAUNCH = "PreLaunch"
    POST_LAUNCH = "PostLaunch"
    SEND_OUTPUT = "SendOutput"
    MEMORY_ALLOC = "MemoryAlloc"
    MEMORY_FREE = "MemoryFree"
    COPY_DATA = "CopyData"
    STREAM_SYNC = "StreamSync"
    PROCESS_MULTI_STREAM = "ProcessMultiStream"
    WAIT_KERNELS_INFER_FINISH = "WaitKernelsInferFinish"
    WAIT_KERNELS_RESIZE_FINISH = "WaitKernelsResizeFinish"
    WAIT_KERNELS_LAUNCH_FINISH = "WaitKernelsLaunchFinish"
    # Inner event is not counted in the total time.
    KERNEL_INFER_INNER = "KernelInferInner"
    KERNEL_INFER_DATA_SYNC = "KernelInferDataSync"
    KERNEL_RESIZE_INNER = "KernelResizeInner"
    KERNEL_LAUNCH_INNER = "KernelLaunchInner"
    BACKEND_GRAPH_RUN_INNER = "BackendGraphRunInner"
    # PyNative Pipeline
    RUN_OP = "RunOp"
    PYNATIVE_FRONTEND_TASK = "PyNativeFrontendTask"
    PYNATIVE_BACKEND_TASK = "PyNativeBackendTask"
    PYNATIVE_DEVICE_TASK = "PyNativeDeviceTask"
    PYNATIVE_LAUNCH_TASK = "PyNativeLaunchTask"
    PYNATIVE_BPROP_TASK = "PyNativeBpropTask"
    WAIT = "Wait"
    # PyNative inner Event
    PYNATIVE_GIL_ACQUIRE = "PyNativeGilAcquire"
    PYNATIVE_CAST = "PyNativeCast"
    PYNATIVE_INFER = "PyNativeInfer"
    PYNATIVE_OP_COMPILE = "PyNativeOpCompile"
    PYNATIVE_GRAD_EXPANDER = "PyNativeGradExpander"
    PYNATIVE_GRAD_UPDATE_SENS = "PyNativeGradUpdateSens"
    PYNATIVE_GRAD_CLEAR_TOP_CELL = "PyNativeGradClearTopCell"
    PYNATIVE_GRAD_CLEAR_AUTO_GRAD_CELL = "PyNativeGradClearAutoGradCell"
    # PyBoost
    PYBOOST_INFER_OUTPUT = "PyBoostInferOutput"
    PYBOOST_INFER_BY_OP_DEF = "PyBoostInferByOpDef"
    PYBOOST_CREATE_OUTPUT_TENSOR = "PyBoostCreateOutputTensor"
    PYBOOST_DEVICE_TASK = "PyBoostDeviceTask"
    PYBOOST_MALLOC_INPUT = "PyBoostMallocInput"
    PYBOOST_MALLOC_OUTPUT = "PyBoostMallocOutput"
    PYBOOST_LAUNCH_ACLLNN = "PyBoostLaunchAclnn"
    PYBOOST_LAUNCH_ATB = "PyBoostLaunchAtb"
    # pybind api
    PYNATIVE_NEW_GRAPH = "PyNativeNewGraph"
    PYNATIVE_END_GRAPH = "PyNativeEndGraph"
    # Python
    PYTHON_OBSERVED = "PythonObserved"
    # Capture Event
    CAPTURE_RUN_GRAPH = "CaptureRunGraph"
    CAPTURE_PROCESS = "CaptureProcess"
    CAPTURE_COMPILE = "CaptureCompile"
    CAPTURE_GUARD = "CaptureGuard"
    # AclNN
    ACLNN_HIT_CACHE_STAGE_1 = "AclnnHitCacheStage1"
    ACLNN_HIT_CACHE_STAGE_2 = "AclnnHitCacheStage2"
    ACLNN_MISS_CACHE_STAGE_1 = "AclnnMissCacheStage1"
    ACLNN_MISS_CACHE_STAGE_2 = "AclnnMissCacheStage2"
    ACLNN_UPDATE_ADDRESS = "AclnnUpdateAddress"
    ACLNN_RUN_OP = "AclnnRunOp"
    # NoGraph grad
    RUN_EXPANDER_FUNC = "RunExpanderFunc"
    EMIT_OP = "EmitOp"
    EXECUTE = "Execute"
    RELEASE_RESOURCE = "ReleaseResource"
    NATIVE_FUNC = "NativeFunc"


class FwkFixSizeFormat:
    """Format definition for framework fixed-size data."""

    OpRangeStruct = "<5Qi3Hb3?"


class OpRangeStructField(Enum):
    """Field indices in operator range structure fixed-size data."""

    THREAD_ID = 0
    FLOW_ID = 1
    STEP = 2
    START_TIME_NS = 3
    END_TIME_NS = 4
    PROCESS_ID = 5
    MODULE_INDEX = 6
    EVENT_INDEX = 7
    STAGE_INDEX = 8
    LEVEL = 9
    IS_GRAPH_DATA = 10
    IS_STAGE = 11
    IS_STACK = 12
    NAME = 13
    FULL_NAME = 14
    MODULE_GRAPH = 15
    EVENT_GRAPH = 16
    CUSTOM_INFO = 17


class FwkProfileDataField:
    """Framework profile data field."""

    @staticmethod
    def _get_enum_value(enum_class, index: int, enum_type: str) -> str:
        """
        Get enum value by index.
        Args:
            enum_class: The enum class to get value from.
            index: The index of the enum value.
            enum_type: The type name of enum for logging.
        Returns:
            The enum value string.
        """
        try:
            # pylint: disable=protected-access
            name = enum_class._member_names_[index]
            return enum_class[name].value
        except IndexError:
            logger.warning(f"Invalid {enum_type} index: {index}")
            return enum_class.DEFAULT.value

    @staticmethod
    def get_stage_value(index: int) -> str:
        """Get stage value."""
        return FwkProfileDataField._get_enum_value(ProfilerStage, index, "stage")

    @staticmethod
    def get_module_value(index: int) -> str:
        """Get module value."""
        return FwkProfileDataField._get_enum_value(ProfilerModule, index, "module")

    @staticmethod
    def get_event_value(index: int) -> str:
        """Get event value."""
        return FwkProfileDataField._get_enum_value(ProfilerEvent, index, "event")


class FwkEventMixin:
    """Mixin class for common framework event functionality."""

    def get_name(self) -> str:
        """Get operator name."""
        op_name = self._origin_data.get(OpRangeStructField.NAME.value, "")
        is_stack = self.fix_size_data[OpRangeStructField.IS_STACK.value]
        if is_stack:
            return op_name

        is_graph_data = self.fix_size_data[OpRangeStructField.IS_GRAPH_DATA.value]
        is_stage = self.fix_size_data[OpRangeStructField.IS_STAGE.value]
        full_name = self._origin_data.get(OpRangeStructField.FULL_NAME.value, "")

        name = ""
        if is_graph_data:
            module_graph = self._origin_data.get(OpRangeStructField.MODULE_GRAPH.value, "")
            event_graph = self._origin_data.get(OpRangeStructField.EVENT_GRAPH.value, "")
            name = f"{module_graph}::{event_graph}::{op_name}"
        elif is_stage:
            stage_index = self.fix_size_data[OpRangeStructField.STAGE_INDEX.value]
            name = FwkProfileDataField.get_stage_value(stage_index)
        elif op_name != "flow":
            module_index = self.fix_size_data[OpRangeStructField.MODULE_INDEX.value]
            event_index = self.fix_size_data[OpRangeStructField.EVENT_INDEX.value]
            module_name = FwkProfileDataField.get_module_value(module_index)
            event_name = FwkProfileDataField.get_event_value(event_index)
            name = f"{module_name}::{event_name}::{full_name}"
        else:
            name = full_name
        return name

    def get_custom_info(self) -> str:
        """Get custom information."""
        value = self._origin_data.get(OpRangeStructField.CUSTOM_INFO.value, None)
        if value is None:
            return ""
        pairs = [pair.split(":") for pair in value.split(";") if pair]
        info_dict = {k: v for k, v in pairs[0:2] if len(pairs) >= 2}
        return str(info_dict)


class FwkCompleteEvent(FwkEventMixin, CompleteEvent):
    """Framework complete event with duration."""

    def __init__(self, data: Dict):
        """Initialize framework complete event."""
        super().__init__(data)
        self.fix_size_data = self._origin_data[FileConstant.FIX_SIZE_DATA]
        self._ts_cache = None
        self._te_cache = None
        self._dur_cache = None
        self._name_cache = None
        self._parent: Optional[BaseEvent] = None
        self._children: List[BaseEvent] = []

    @property
    def parent(self) -> BaseEvent:
        """Get parent event."""
        return self._parent

    @parent.setter
    def parent(self, event: BaseEvent) -> None:
        """Set parent event."""
        self._parent = event

    @property
    def children(self) -> List[BaseEvent]:
        """Get child events."""
        return self._children

    @property
    def ts_raw(self) -> int:
        """Get raw start timestamp."""
        return self.fix_size_data[OpRangeStructField.START_TIME_NS.value]

    @property
    def ts(self) -> Decimal:
        """Get start time in us."""
        if not self._ts_cache:
            self._ts_cache = TimeConverter.convert_syscnt_to_timestamp_us(
                self.fix_size_data[OpRangeStructField.START_TIME_NS.value]
            )
        return self._ts_cache

    @property
    def te(self) -> Decimal:
        """Get end time in us."""
        if not self._te_cache:
            self._te_cache = TimeConverter.convert_syscnt_to_timestamp_us(
                self.fix_size_data[OpRangeStructField.END_TIME_NS.value]
            )
        return self._te_cache

    @property
    def dur(self) -> Decimal:
        """Get duration in us."""
        if not self._dur_cache:
            self._dur_cache = self.te - self.ts
        return self._dur_cache

    @property
    def pid(self) -> int:
        """Get process ID."""
        return int(EventConstant.MINDSPORE_PID)

    @property
    def tid(self) -> int:
        """Get thread ID."""
        return int(self.fix_size_data[OpRangeStructField.THREAD_ID.value])

    @property
    def id(self) -> int:
        """Get event ID."""
        return int(self.fix_size_data[OpRangeStructField.FLOW_ID.value])

    @property
    def name(self) -> str:
        """Get operator name."""
        if not self._name_cache:
            self._name_cache = self.get_name()
        return self._name_cache

    @property
    def step(self) -> int:
        """Get step ID."""
        return int(self.fix_size_data[OpRangeStructField.STEP.value])

    @property
    def is_stack(self) -> bool:
        """Get is stack."""
        return bool(self.fix_size_data[OpRangeStructField.IS_STACK.value])

    @property
    def cat(self) -> str:
        """Get event category."""
        return EventConstant.STACK_EVENT_CAT if self.is_stack else ""

    @property
    def level(self) -> int:
        """Get event level."""
        return int(self.fix_size_data[OpRangeStructField.LEVEL.value])

    @property
    def custom_info(self) -> str:
        """Get custom information."""
        return self.get_custom_info()


class FwkInstantEvent(FwkEventMixin, InstantEvent):
    """Framework instant event without duration."""

    def __init__(self, data: Dict):
        """Initialize framework instant event."""
        super().__init__(data)
        self.fix_size_data = self._origin_data[FileConstant.FIX_SIZE_DATA]
        self._ts_cache = None
        self._name_cache = None

    @property
    def ts_raw(self) -> int:
        """Get raw start timestamp."""
        return self.fix_size_data[OpRangeStructField.START_TIME_NS.value]

    @property
    def ts(self) -> Decimal:
        """Get time in us."""
        if not self._ts_cache:
            self._ts_cache = TimeConverter.convert_syscnt_to_timestamp_us(
                self.fix_size_data[OpRangeStructField.START_TIME_NS.value]
            )
        return self._ts_cache

    @property
    def pid(self) -> int:
        """Get process ID."""
        return int(EventConstant.MINDSPORE_PID)

    @property
    def tid(self) -> int:
        """Get thread ID."""
        return int(self.fix_size_data[OpRangeStructField.THREAD_ID.value])

    @property
    def name(self) -> str:
        """Get operator name."""
        if not self._name_cache:
            self._name_cache = self.get_name()
        return self._name_cache

    @property
    def step(self) -> int:
        """Get step ID."""
        return int(self.fix_size_data[OpRangeStructField.STEP.value])

    @property
    def level(self) -> int:
        """Get event level."""
        return int(self.fix_size_data[OpRangeStructField.LEVEL.value])

    @property
    def custom_info(self) -> str:
        """Get custom information."""
        return self.get_custom_info()


class FwkMetaEvent(MetaEvent):
    """Framework metadata event."""

    @property
    def pid(self) -> int:
        """Get framework process ID."""
        return int(EventConstant.MINDSPORE_PID)
