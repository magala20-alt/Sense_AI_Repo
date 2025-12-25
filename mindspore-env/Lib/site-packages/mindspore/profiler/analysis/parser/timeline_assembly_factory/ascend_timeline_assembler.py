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
"""Timeline assembler for Ascend device."""
from typing import List, Dict, Any
from decimal import Decimal
from collections import defaultdict

from mindspore import log as logger
from mindspore.profiler.common.log import ProfilerLogger
from mindspore.profiler.common.constant import EventConstant, TimelineLayerName, ProfilerLevel, JitLevel
from mindspore.profiler.analysis.parser.timeline_event.base_event import BaseEvent
from mindspore.profiler.analysis.parser.timeline_event.timeline_event_pool import TimelineEventPool
from mindspore.profiler.analysis.parser.timeline_event.flow_event import FlowStartEvent, FlowEndEvent
from mindspore.profiler.analysis.parser.timeline_creator.fwk_timeline_creator import FwkTimelineCreator
from mindspore.profiler.analysis.parser.timeline_creator.cpu_op_timeline_creator import CpuOpTimelineCreator
from mindspore.profiler.analysis.parser.timeline_creator.msprof_timeline_creator import MsprofTimelineCreator
from mindspore.profiler.analysis.parser.timeline_assembly_factory.base_timeline_assembler import BaseTimelineAssembler
from mindspore.profiler.analysis.parser.timeline_creator.scope_layer_timeline_creator import (
    ScopeLayerTimelineCreator,
    is_scope_data
)


class AscendTimelineAssembler(BaseTimelineAssembler):
    """Assembler for Ascend device timeline."""

    def __init__(self, **kwargs):
        super().__init__()
        self._profiler_level = kwargs.get("profiler_level")
        self._jit_level = kwargs.get("jit_level")
        self._init_creators()
        ProfilerLogger.init(kwargs.get("ascend_ms_dir"))
        self._logger = ProfilerLogger.get_instance()

    def _init_creators(self):
        """Initialize trace creators."""
        self._fwk_creator = FwkTimelineCreator()
        self._cpu_op_creator = CpuOpTimelineCreator()
        self._msprof_creator = MsprofTimelineCreator()
        self._scope_layer_creator = ScopeLayerTimelineCreator()

    def assemble(self, data: Dict[str, Any]) -> None:
        """Assemble Ascend timeline from input data."""
        self._assemble_basic_events(data)
        self._assemble_flow_events()
        self._assemble_scope_layer_events()

    def _assemble_basic_events(self, data: Dict[str, Any]) -> None:
        """Create and add basic events from input data."""
        self._assemble_events(self._fwk_creator, data.get("mindspore_op_list", []))
        self._assemble_events(self._cpu_op_creator, data.get("cpu_op_lines", []))
        self._assemble_events(self._msprof_creator, data.get("msprof_timeline", []))

    def _assemble_scope_layer_events(self) -> None:
        """Create scope layer events."""
        scope_data = []

        # Get CPU OP scope data if available
        cpu_op_pool = self.trace_view_container.get_pool_by_name(TimelineLayerName.CPU_OP.value)
        if cpu_op_pool:
            for event in cpu_op_pool.get_complete_events():
                if is_scope_data(event):
                    scope_data.append(event)

        # Get Ascend Hardware scope data
        hardware_pool = self.trace_view_container.get_pool_by_name(TimelineLayerName.ASCEND_HARDWARE.value)
        if hardware_pool:
            for event in hardware_pool.get_complete_events():
                if is_scope_data(event) or is_scope_data(event.parent):
                    scope_data.append(event)

        if scope_data:
            self._assemble_events(self._scope_layer_creator, scope_data)

    def _assemble_events(self, creator, data) -> None:
        """Create events using creator and add to container."""
        creator.create(data)
        for pool in creator.get_event_pools().values():
            self.trace_view_container.add_event_pool(pool)
        self.trace_view_container.add_trace_events(creator.get_chrome_trace_data())

    def _assemble_flow_events(self) -> None:
        """Create and add flow events between timelines."""
        fwk_pool = self.trace_view_container.get_pool_by_name(TimelineLayerName.MINDSPORE.value)
        if not fwk_pool:
            return

        # Create and add fwk to fwk flows
        fwk_to_fwk_flows = self._create_fwk_to_fwk_flow(fwk_pool)
        self.trace_view_container.add_trace_events(fwk_to_fwk_flows)

        # Create and add fwk to mstx flows
        for mstx_name in TimelineLayerName.MSTX.value:
            mstx_pool = self.trace_view_container.get_pool_by_name(mstx_name)
            if mstx_pool:
                fwk_to_mstx_flows = self._create_fwk_to_mstx_flow(mstx_pool, fwk_pool)
                self.trace_view_container.add_trace_events(fwk_to_mstx_flows)

        if self._profiler_level == ProfilerLevel.LevelNone.value:
            return

        hardware_pool = self.trace_view_container.get_pool_by_name(TimelineLayerName.ASCEND_HARDWARE.value)
        cann_pool = self.trace_view_container.get_pool_by_name(TimelineLayerName.CANN.value)
        if not hardware_pool or not cann_pool:
            return

        # Collect kernel launch events
        for event in fwk_pool.get_complete_events():
            if any(keyword in event.name for keyword in EventConstant.KERNEL_LAUNCH_KEYWORDS):
                self.trace_view_container.kernel_launch_op_event[event.tid].append(event)

        # Create and add fwk to hardware flows
        fwk_to_hardware_flows = self._create_fwk_to_hardware_flow()
        self.trace_view_container.add_trace_events(fwk_to_hardware_flows)

    def _create_fwk_to_hardware_flow(self) -> List[Dict]:
        """Create flow events between framework and hardware events."""
        acl_to_npu_flow_dict = self._msprof_creator.get_acl_to_npu_flow_dict()
        fwk_launch_op_list = self.trace_view_container.kernel_launch_op_event
        # The GE backend does not have the flow from CANN to hardware at each step
        if not acl_to_npu_flow_dict and self._jit_level != JitLevel.GRAPH_LEVEL:
            logger.error("Cannot find connection between CANN layer and Ascend Hardware layer.")
            return []
        # The GE backend does not have "KernelLaunch" or "LaunchTask" keywords
        if not fwk_launch_op_list and self._jit_level != JitLevel.GRAPH_LEVEL:
            logger.warning("Cannot find launch op in MindSpore framework.")
            return []
        if set(acl_to_npu_flow_dict.keys()) != set(fwk_launch_op_list.keys()):
            self._logger.warning(
                "The number of launch op threads in MindSpore framework is inconsistent with the CANN layer.")

        fwk_to_npu_flows = []
        for tid, cann_to_npu_events in acl_to_npu_flow_dict.items():
            fwk_launch_op_sorted = sorted(fwk_launch_op_list.get(tid, []), key=lambda x: x.ts)
            acl_to_npu_events_sorted = sorted(cann_to_npu_events.items(), key=lambda x: Decimal(x[0]))

            index = 0
            for acl_start_time, device_data_list in acl_to_npu_events_sorted:
                acl_start_time = Decimal(acl_start_time)
                while index < len(fwk_launch_op_sorted):
                    fwk_launch_op = fwk_launch_op_sorted[index]
                    if fwk_launch_op.ts > acl_start_time:
                        break
                    if acl_start_time <= fwk_launch_op.te:
                        for hardware_event in device_data_list:
                            hardware_event.parent = fwk_launch_op
                            fwk_launch_op.children.append(hardware_event)
                            self.trace_view_container.hardware_op_event[hardware_event.tid].append(hardware_event)
                            fwk_to_npu_flows.extend(
                                self._create_flow_events(
                                    fwk_launch_op,
                                    hardware_event,
                                    EventConstant.MINDSPORE_NPU_FLOW_NAME,
                                    EventConstant.MINDSPORE_NPU_FLOW_CAT
                                )
                            )
                        break
                    index += 1

        return fwk_to_npu_flows

    def _create_fwk_to_fwk_flow(self, framework_pool: TimelineEventPool) -> List[Dict]:
        """Create flow events between framework events."""
        fwk_to_fwk_flows = []
        for flow_id, flow_pair in framework_pool.get_start_to_end_flow_pairs().items():
            if len(flow_pair["start"]) != 1 or len(flow_pair["end"]) != 1:
                logger.info(
                    f"Mindspore op flow expected exactly one start and one end event with flow id {flow_id}, "
                    f"but got {len(flow_pair['start'])} start and {len(flow_pair['end'])} end events"
                )
                continue

            start_event = flow_pair["start"][0]
            end_event = flow_pair["end"][0]
            end_event.parent = start_event
            start_event.children.append(end_event)

            fwk_to_fwk_flows.extend(
                self._create_flow_events(
                    start_event,
                    end_event,
                    EventConstant.MINDSPORE_SELF_FLOW_NAME,
                    EventConstant.MINDSPORE_SELF_FLOW_CAT,
                    flow_id
                )
            )

        return fwk_to_fwk_flows

    def _create_fwk_to_mstx_flow(self, mstx_pool: TimelineEventPool, fwk_pool: TimelineEventPool) -> List[Dict]:
        """Create flow events between framework and mstx events."""
        fwk_mstx_api_event_group_by_tid = defaultdict(list)
        for event in fwk_pool.get_complete_events():
            if EventConstant.MSTX_KEYWORD in event.name:
                fwk_mstx_api_event_group_by_tid[event.tid].append(event)

        fwk_to_mstx_flows = []
        mstx_event_group_by_tid = mstx_pool.complete_event

        for tid, mstx_event_list in mstx_event_group_by_tid.items():
            sorted_fwk_mstx_api_events = sorted(fwk_mstx_api_event_group_by_tid.get(tid, []), key=lambda x: x.ts)
            sorted_mstx_events = sorted(mstx_event_list, key=lambda x: x.ts)

            index = 0
            for mstx_event in sorted_mstx_events:
                while index < len(sorted_fwk_mstx_api_events):
                    fwk_event = sorted_fwk_mstx_api_events[index]
                    if mstx_event.ts < fwk_event.ts:
                        break
                    if mstx_event.ts <= fwk_event.te:
                        mstx_event.parent = fwk_event
                        fwk_event.children.append(mstx_event)
                        fwk_to_mstx_flows.extend(
                            self._create_flow_events(
                                fwk_event,
                                mstx_event,
                                EventConstant.MSTX_FLOW_NAME,
                                EventConstant.MSTX_FLOW_CAT,
                            )
                        )
                        index += 1
                        break
                    index += 1

        return fwk_to_mstx_flows

    @staticmethod
    def _create_flow_events(start_event: BaseEvent, end_event: BaseEvent,
                            name: str, cat: str, flow_id: str = None) -> List[Dict]:
        """Create flow start and end events pair."""
        if flow_id is None:
            flow_id = str(end_event.ts)

        flow_start = FlowStartEvent({
            "name": name,
            "cat": cat,
            "pid": start_event.pid,
            "tid": start_event.tid,
            "ts": start_event.ts,
            "id": flow_id,
            "bp": "e"
        })
        flow_end = FlowEndEvent({
            "name": name,
            "cat": cat,
            "pid": end_event.pid,
            "tid": end_event.tid,
            "ts": end_event.ts,
            "id": flow_id,
            "bp": "e"
        })
        return [flow_start.to_trace_format(), flow_end.to_trace_format()]
