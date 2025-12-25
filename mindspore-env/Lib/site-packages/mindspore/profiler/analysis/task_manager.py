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
"""Task manager"""
import time
from typing import Dict, Any
from collections import defaultdict
from multiprocessing import Manager
from multiprocessing import Process

from mindspore import log as logger
from mindspore.profiler.common.process_bar import ProcessBar
from mindspore.profiler.analysis.parser.base_parser import BaseParser
from mindspore.profiler.analysis.work_flow import WorkFlow
from mindspore.profiler.common.log import ProfilerLogger


class TaskManager:
    """
    Manages the execution of workflows consisting of multiple parsers.
    """

    ROUND_DECIMAL = 2

    def __init__(self):
        """
        Initialize the TaskManager with empty workflows and cost time tracking.
        """
        self.workflows: Dict[str, WorkFlow] = defaultdict(WorkFlow)
        self.show_process: Dict[str, bool] = defaultdict(bool)
        self.flows_cost_time: Dict[str, Dict[str, Any]] = Manager().dict()
        self._logger = ProfilerLogger.get_instance()

    @property
    def cost_time(self) -> Dict[str, Dict[str, Any]]:
        # convert Manager().dict() to dict for json serialization
        return dict(self.flows_cost_time)

    def create_flow(self, *parsers: BaseParser, flow_name: str, show_process: bool = False) -> None:
        """
        Create a workflow with a list of parsers.

        Args:
            *parsers (BaseParser): The parsers to be executed in the workflow.
            flow_name (str): The name of the workflow.
            show_process (bool): Whether to show the process bar of the workflow.

        Raises:
            ValueError: If any of the provided parsers is not an instance of BaseParser.
        """
        if not parsers:
            logger.error("No parsers provided")
            return

        workflow = WorkFlow()
        for parser in parsers:
            if not isinstance(parser, BaseParser):
                raise ValueError(
                    f"parser {parser.__class__.__name__} must be a BaseParser"
                )
            workflow.add_parser(parser)

        self.workflows[flow_name] = workflow
        self.show_process[flow_name] = show_process

    def run(self) -> None:
        """
        Run all workflows with the given data using a ProcessPoolExecutor.

        Args:
            data (Any): The data to be processed by the workflows.
        """
        processes = []
        for flow_name, workflow in self.workflows.items():
            p = Process(target=self._run_flow, args=(flow_name, workflow))
            processes.append((p, flow_name))
            p.start()
            self._logger.info("TaskManager run flow [%s] [pid: %s] start", flow_name, p.pid)

        for p, flow_name in processes:
            p.join()
            self._logger.info("TaskManager flow [%s] [pid: %s] join", flow_name, p.pid)

    def _run_flow(self, flow_name: str, workflow: WorkFlow) -> None:
        """
        Run a single workflow with the given data.

        Args:
            flow_name (str): The name of the workflow.
            workflow (WorkFlow): The workflow to be executed.
            data (Any): The data to be processed by the workflow.
        """
        start_time = time.perf_counter()
        parser_cost_time = defaultdict(float)

        parsers = (
            ProcessBar(workflow, desc="Parsing")
            if self.show_process[flow_name]
            else workflow
        )

        data = {}
        try:
            for parser in parsers:
                parser_start_time = time.perf_counter()
                data = parser.parse(data)
                parser_end_time = time.perf_counter()
                parser_cost_time[parser.__class__.__name__] = round(
                    parser_end_time - parser_start_time, self.ROUND_DECIMAL
                )
        except Exception as e: # pylint: disable=W0703
            logger.error("Parser %s error: %s", parser.__class__.__name__, str(e))
            self._logger.error("TaskManager run [%s] error: %s", flow_name, str(e), exc_info=True)

        end_time = time.perf_counter()
        # Record the cost time of the workflow
        self.flows_cost_time[flow_name] = {
            "total_time_seconds": round(end_time - start_time, self.ROUND_DECIMAL),
            "parser_times_seconds": parser_cost_time,
        }
