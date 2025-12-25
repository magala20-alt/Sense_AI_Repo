# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Base class for timeline assembly process."""
from abc import ABC, abstractmethod
from typing import Dict, Any

from mindspore.profiler.analysis.parser.timeline_assembly_factory.trace_view_container import TraceViewContainer


class BaseTimelineAssembler(ABC):
    """Base class for timeline assembly.

    This class defines the basic interface for timeline assembly process. It coordinates
    different trace event pools and manages the assembly of the final timeline view.
    """

    def __init__(self):
        """Initialize timeline assembler."""
        self.trace_view_container = TraceViewContainer()

    @abstractmethod
    def assemble(self, data: Dict[str, Any]) -> None:
        """Assemble timeline from input data.

        Args:
            data (Dict[str, Any]): Input data containing various timeline information.
        """

    def get_trace_view_container(self) -> TraceViewContainer:
        """Get the trace container object."""
        return self.trace_view_container
