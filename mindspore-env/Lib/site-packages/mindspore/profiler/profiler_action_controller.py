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
# ============================================================================
"""Profiler Action Controller"""
__all__ = []

from functools import partial
from typing import Optional, Callable, Any, Dict

from mindspore.profiler.profiler_interface import ProfilerInterface
from mindspore.profiler.schedule import ProfilerAction
from mindspore import log as logger


class ProfilerActionController:
    """
    A controller class for managing profiler actions and transitions.

    This class handles the actions and transitions between different profiler states.
    It uses an action_map to determine the actions to take based on the previous and
    current profiler actions.

    Attributes:
        prof_interface (ProfilerInterface): The profiler interface instance.
        on_trace_ready (Optional[Callable[..., Any]]): A callback function to be called when the trace is ready.
        action_map (Dict): A map of state transitions and their corresponding actions.
    """

    def __init__(self, prof_interface: ProfilerInterface, on_trace_ready: Optional[Callable[..., Any]] = None) -> None:
        """
        Initializes a new instance of ProfilerActionController.

        Args:
            prof_interface (ProfilerInterface): The profiler interface instance.
            on_trace_ready (Optional[Callable[..., Any]]): A callback function to be called when the trace is ready.
        """
        self.prof_interface = prof_interface
        self.on_trace_ready = on_trace_ready
        self.action_map = self._init_action_map()

    def _trace_ready(self):
        """
        Calls the on_trace_ready callback function if it is set.

        This method is called when the trace is ready to notify the callback function.
        """
        if self.on_trace_ready:
            self.on_trace_ready(self.prof_interface)

    def transit_action(self, prev_action: ProfilerAction, current_action: ProfilerAction) -> None:
        """
        Handles actions between previous action and current action

        Args:
            prev_action: The previous state
            current_action: the current state
        """
        # Get the action list for this state transition
        action_list = self.action_map.get((prev_action, current_action))

        if action_list:
            logger.info(f"ProfilerAction transition: {prev_action} -> {current_action}")
            for action in action_list:
                action()

    def _init_action_map(self) -> Dict:
        """
        Initialize the action map for state transitions.

        Returns:
            Dict: A map of state transitions and their corresponding actions.
        """
        action_map = {
            (ProfilerAction.NONE, ProfilerAction.NONE): [],
            (ProfilerAction.NONE, ProfilerAction.WARM_UP): [self.prof_interface.init],
            (ProfilerAction.NONE, ProfilerAction.RECORD): [self.prof_interface.init, self.prof_interface.start],
            (ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE): [self.prof_interface.init,
                                                                    self.prof_interface.start],

            (ProfilerAction.WARM_UP, ProfilerAction.NONE): [
                partial(logger.warning, "Incorrect schedule: WARMUP followed by NONE"),
                self.prof_interface.start,
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self.prof_interface.clear
            ],
            (ProfilerAction.WARM_UP, ProfilerAction.WARM_UP): [],
            (ProfilerAction.WARM_UP, ProfilerAction.RECORD): [self.prof_interface.start],
            (ProfilerAction.WARM_UP, ProfilerAction.RECORD_AND_SAVE): [self.prof_interface.start],

            (ProfilerAction.RECORD, ProfilerAction.NONE): [
                partial(logger.warning, "Incorrect schedule: RECORD followed by NONE"),
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self.prof_interface.clear
            ],
            (ProfilerAction.RECORD, ProfilerAction.WARM_UP): [
                partial(logger.warning, "Incorrect schedule: RECORD followed by WARMUP"),
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self.prof_interface.clear
            ],
            (ProfilerAction.RECORD, ProfilerAction.RECORD): [],
            (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE): [],

            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE): [
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self._trace_ready,
                self.prof_interface.clear
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.WARM_UP): [
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self._trace_ready,
                self.prof_interface.clear,
                self.prof_interface.init,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD): [
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self._trace_ready,
                self.prof_interface.clear,
                self.prof_interface.init,
                self.prof_interface.start
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD_AND_SAVE): [
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self._trace_ready,
                self.prof_interface.clear,
                self.prof_interface.init,
                self.prof_interface.start
            ],

            # Used for exit action
            (ProfilerAction.WARM_UP, None): [
                partial(logger.warning,
                        "Incorrect schedule: Stop profiler while current state is WARMUP "
                        "which will result in empty parsed data."),
                self.prof_interface.finalize,
                self.prof_interface.clear,
                self.prof_interface.delete_dir
            ],
            (ProfilerAction.RECORD, None): [
                partial(logger.warning,
                        "Incorrect schedule: Stop profiler while current state is RECORD "
                        "which may result in incomplete parsed data."),
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self._trace_ready,
                self.prof_interface.clear
            ],
            (ProfilerAction.RECORD_AND_SAVE, None): [
                partial(logger.warning,
                        "Stop profiler while current state is RECORD_AND_SAVE, "
                        "perhaps the scheduling cycle has not yet completed."),
                self.prof_interface.stop,
                self.prof_interface.finalize,
                self._trace_ready,
                self.prof_interface.clear
            ]
        }
        return action_map
