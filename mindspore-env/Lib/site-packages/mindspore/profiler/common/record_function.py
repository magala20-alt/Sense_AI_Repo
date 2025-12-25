# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
# ====================
"""Record Function"""

from mindspore._c_expression import PythonProfilerRecorder


class RecordFunction:
    """
    A context manager for recording profiling data using PythonProfilerRecorder.

    This class provides a convenient way to start and stop recording profiling data
    using a PythonProfilerRecorder instance. It can be used as a context manager to
    ensure that recording is properly started and stopped, even if an exception occurs.

    Attributes:
        recorder (PythonProfilerRecorder): The underlying profiler recorder instance.

    Methods:
        start(): Starts the recording process.
        stop(): Stops the recording process.
        __enter__(): Starts the recording process when entering a with statement.
        __exit__(exc_type): Stops the recording process when exiting a with statement.
    """

    def __init__(self, name):
        """
        Initializes a new instance of RecordFunction.

        Args:
            name (str): The name of the profiling record.
        """
        self.recorder = PythonProfilerRecorder(name)

    def start(self):
        """
        Starts the recording process.

        """
        self.recorder.record_start()

    def stop(self):
        """
        Stops the recording process.

        """
        self.recorder.record_end()

    def __enter__(self) -> None:
        """
        Starts the recording process when entering a with statement.

        """
        self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Stops the recording process when exiting a with statement.

        Args:
            exc_type (type): The type of exception that occurred, if any.

        """
        self.stop()
