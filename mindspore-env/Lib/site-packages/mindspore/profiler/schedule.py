# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
"""Profiler Schedule"""
__all__ = ["ProfilerAction", "Schedule"]

from enum import Enum

from mindspore import log as logger


class ProfilerAction(Enum):
    """
    Enum class representing different actions that can be performed by the profiler.

    Each member of the enum represents a specific profiling action, which can be used
    to control the behavior of the profiler at different stages of execution.

    Attributes:
        NONE (ProfilerAction): No profiling action.
        WARM_UP (ProfilerAction): Warm-up phase of profiling.
        RECORD (ProfilerAction): Record phase of profiling.
        RECORD_AND_SAVE (ProfilerAction): Record and save phase of profiling.
    """
    NONE = 0
    WARM_UP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3

    @staticmethod
    def get_by_value(value):
        """
        Retrieves a ProfilerAction enum member by its value.

        Args:
            value (int): The value of the ProfilerAction enum member to retrieve.

        Returns:
            ProfilerAction, The enum member corresponding to the given value, or None if not found.
        """
        value_map = {action.value: action for action in ProfilerAction}
        return value_map.get(value, None)


class Schedule:
    r"""
    This class use to get the actions of each step.
    The schedule is as follows:

    .. code-block::

        (NONE)        (NONE)          (NONE)       (WARM_UP)       (RECORD)      (RECORD)     (RECORD_AND_SAVE)    None
        START------->skip_first------->wait-------->warmup-------->active........active.........active----------->stop
                                      |                                                             |
                                      |                           repeat_1                          |
                                      ---------------------------------------------------------------

    The profiler will skip the first ``skip_first`` steps, then wait for ``wait`` steps,
    then do the warmup for the next ``warmup`` steps, then do the active recording for the next
    ``active`` steps and then repeat the cycle starting with ``wait`` steps. The optional number
    of cycles is specified with the ``repeat`` parameter, the zero value means that
    the cycles will continue until the profiling is finished.

    Keyword Args:
        wait (int): The number of steps to wait before starting the warm-up phase.
            must be greater than or equal to 0. If the wait parameter is not set externally,
            it is set to ``0`` when the schedule class is initialized.
        active (int): The number of steps to record data during the active phase.
            must be greater than or equal to 1. If the active parameter is not set externally,
            it is set to ``1`` when the schedule class is initialized.
        warmup (int, optional): The number of steps to perform the warm-up phase.
            must be greater than or equal to 0. Default value: ``0``.
        repeat (int, optional): The number of times to repeat the cycle.
            If repeat is set to 0, the Profiler will determine the repeat value based on the number of times the model
            is trained, for example, if the total training steps are 100, wait+active+warmup=10, skip_first=10,
            Then repeat=(100-10)/10=9, indicating that the execution is repeated 9 timeswhich will
            generate one more performance data with incomplete collection. The data in the last step is abnormal data
            that users do not need to pay attention to. Suggest configuring integers greater than 0. When using
            cluster analysis tools or MindStudio Insight to view, it is recommended to configure it as 1;
            If the setting is greater than 1, the collected performance data folder needs to be divided into repeat and
            other parts, placed in different folders for re-parsing, and classified according to the timestamp order in
            the folder name. Default value: ``0``.
        skip_first (int, optional): The number of steps to skip at the beginning. Must be greater than or equal to 0.
            Default value: ``0``

    Raises:
        ValueError: When the parameter step is less than 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> import mindspore.dataset as ds
        >>> from mindspore import context, nn
        >>> from mindspore.profiler import ProfilerLevel, AicoreMetrics, ExportType, ProfilerActivity
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.fc = nn.Dense(2, 2)
        ...
        ...     def construct(self, x):
        ...         return self.fc(x)
        >>>
        >>> def generator_net():
        ...     for _ in range(2):
        ...         yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)
        >>>
        >>> def train(test_net):
        ...     optimizer = nn.Momentum(test_net.trainable_params(), 1, 0.9)
        ...     loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        ...     data = ds.GeneratorDataset(generator_net(), ["data", "label"])
        ...     model = mindspore.train.Model(test_net, loss, optimizer)
        ...     model.train(1, data)
        >>>
        >>> if __name__ == '__main__':
        ...     # If the device_target is GPU, set the device_target to "GPU"
        ...     context.set_context(mode=mindspore.GRAPH_MODE)
        ...     mindspore.set_device("Ascend")
        ...
        ...     # Init Profiler
        ...     experimental_config = mindspore.profiler._ExperimentalConfig(
        ...                                 profiler_level=ProfilerLevel.Level0,
        ...                                 aic_metrics=AicoreMetrics.AiCoreNone,
        ...                                 l2_cache=False,
        ...                                 mstx=False,
        ...                                 data_simplification=False,
        ...                                 export_type=[ExportType.Text])
        ...     steps = 10
        ...     net = Net()
        ...     # Note that the Profiler should be initialized before model.train
        ...     with mindspore.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        ...                                     schedule=mindspore.profiler.schedule(wait=1, warmup=1, active=2,
        ...                                           repeat=1, skip_first=2),
        ...                                     on_trace_ready=mindspore.profiler.tensorboard_trace_handler("./data"),
        ...                                     profile_memory=False,
        ...                                     experimental_config=experimental_config) as prof:
        ...
        ...         # Train Model
        ...         for step in range(steps):
        ...             train(net)
        ...             prof.step()
    """

    def __init__(self, *, wait: int, active: int, warmup: int = 0, repeat: int = 0, skip_first: int = 0) -> None:
        self.wait = wait
        self.active = active
        self.warmup = warmup
        self.repeat = repeat
        self.skip_first = skip_first
        self._check_params()

    def __call__(self, step: int) -> ProfilerAction:
        """
        Obtain the action of the specified step from the schedule.

        Args:
            step (int): step num.

        Returns:
            ProfilerAction, The action corresponding to a step.
        """
        if step < 0:
            raise ValueError("Invalid parameter step, which must be not less than 0.")
        if step < self.skip_first:
            return ProfilerAction.NONE

        step -= self.skip_first

        num_steps = self.wait + self.warmup + self.active
        if 0 < self.repeat <= step / num_steps:
            return ProfilerAction.NONE

        mod_step = step % num_steps
        if mod_step < self.wait:
            return ProfilerAction.NONE
        if mod_step < self.wait + self.warmup:
            return ProfilerAction.WARM_UP
        return ProfilerAction.RECORD if mod_step < num_steps - 1 else ProfilerAction.RECORD_AND_SAVE

    def __repr__(self):
        return (f"Schedule(wait={self.wait!r}, active={self.active!r}, "
                f"warmup={self.warmup!r}, repeat={self.repeat!r}, "
                f"skip_first={self.skip_first!r})")

    def _check_params(self):
        """
        Verify all parameters in the schedule,
        and set them to default values if the parameters are not compliant.
        """
        if not isinstance(self.wait, int) or isinstance(self.wait, bool) or self.wait < 0:
            logger.warning(f"Parameter 'wait' should be of type int, but got "
                           f"{type(self.wait).__name__}. reset to int 0.")
            self.wait = 0
        if not isinstance(self.warmup, int) or isinstance(self.warmup, bool) or self.warmup < 0:
            logger.warning(f"Parameter 'warmup' should be of type int, but got "
                           f"{type(self.warmup).__name__}. reset to int 0.")
            self.warmup = 0
        if not isinstance(self.active, int) or isinstance(self.active, bool) or self.active <= 0:
            logger.warning(f"Parameter 'active' should be of type int, but got "
                           f"{type(self.active).__name__}. reset to int 1.")
            self.active = 1
        if not isinstance(self.repeat, int) or isinstance(self.repeat, bool) or self.repeat < 0:
            logger.warning(f"Parameter 'repeat' should be of type int, but got "
                           f"{type(self.repeat).__name__}. reset to int 0.")
            self.repeat = 0
        if not isinstance(self.skip_first, int) or isinstance(self.skip_first, bool) or self.skip_first < 0:
            logger.warning(f"Parameter 'skip_first' should be of type int, but got "
                           f"{type(self.skip_first).__name__}. reset to int 0.")
            self.skip_first = 0
        if self.warmup == 0:
            logger.warning("Profiler won't be using warmup, this can skew profiler results")

    def to_dict(self):
        """
        Convert schedule to a dict.

        Returns:
            dict, the parameters of schedule and their values.
        """
        return {'wait': self.wait, 'active': self.active, 'warmup': self.warmup,
                'repeat': self.repeat, 'skip_first': self.skip_first}


def _default_schedule_fn(_: int) -> ProfilerAction:
    """
    Default profiler behavior - immediately starts recording the events,
    keeps doing it on every profiler step.

    Args:
        _ (int): step num.

    Returns:
        ProfilerAction, The RECORD action.
    """
    return ProfilerAction.RECORD
