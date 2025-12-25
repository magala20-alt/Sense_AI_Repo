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
"""ascend timeline viewer"""
import os
from typing import List, Dict

from mindspore.profiler.analysis.viewer.base_viewer import BaseViewer
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.log import ProfilerLogger


class AscendTimelineViewer(BaseViewer):
    """Ascend Timeline Viewer for analyzing and saving timeline data."""

    _TRACE_VIEW_FILE_NAME = 'trace_view.json'

    def __init__(self, **kwargs):
        """Initialize the AscendTimelineViewer."""
        super().__init__()
        self._save_path = os.path.join(
            kwargs.get("ascend_profiler_output_path"),
            self._TRACE_VIEW_FILE_NAME
        )
        self._ascend_ms_dir = kwargs.get("ascend_ms_dir")
        ProfilerLogger.init(self._ascend_ms_dir)
        self._logger = ProfilerLogger.get_instance()

    def save(self, data: Dict) -> None:
        """Get the input data and save the timeline data."""
        self._logger.info("AscendTimelineViewer start")
        try:
            trace_view_container = data.get("trace_view_container", None)
            if not trace_view_container:
                raise RuntimeError("The trace view container is None, Failed to save trace_view.json.")
            trace_view_data = trace_view_container.get_trace_view()
            self._save_data(trace_view_data)
            self._logger.info("Trace viewer save trace_view.json done")
        except Exception as e: # pylint: disable=W0703
            self._logger.error("Failed to save trace_view.json: %s", e, exc_info=True)
        self._logger.info("AscendTimelineViewer end")

    def _save_data(self, timeline_data: List[Dict]) -> None:
        """Save the timeline data to a JSON file."""
        self._logger.info("Trace view saved start")
        FileManager.create_json_file(self._save_path, timeline_data)
        self._logger.info("Trace view saved done, save path: %s", self._save_path)
