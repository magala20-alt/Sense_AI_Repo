# Copyright 2022-2024 Huawei Technologies Co., Ltd
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
# ===========================================================================
"""ProfilerParameters"""
import warnings
from typing import Dict, Optional, Callable, Any

from mindspore import log as logger
from mindspore.profiler.common.constant import (
    ProfilerLevel,
    ProfilerActivity,
    AicoreMetrics,
    ExportType,
    HostSystem
)
from mindspore.profiler.schedule import Schedule


class ProfilerParameters:
    """
    Profiler parameters manage all parameters, parameters validation and type conversion.
    """

    # key: Parameter name, value: (type, default value)
    PARAMS: Dict[str, tuple] = {
        "output_path": (str, "./data"),
        "profiler_level": (ProfilerLevel, ProfilerLevel.Level0),
        "activities": (list, [ProfilerActivity.CPU, ProfilerActivity.NPU]),
        "aic_metrics": (AicoreMetrics, AicoreMetrics.AiCoreNone),
        "with_stack": (bool, False),
        "profile_memory": (bool, False),
        "data_process": (bool, False),
        "parallel_strategy": (bool, False),
        "start_profile": (bool, True),
        "l2_cache": (bool, False),
        "hbm_ddr": (bool, False),
        "pcie": (bool, False),
        "sys_io": (bool, False),
        "sys_interconnection": (bool, False),
        "host_sys": (list, []),
        "sync_enable": (bool, True),
        "data_simplification": (bool, True),
        "record_shapes": (bool, False),
        "export_type": (list, [ExportType.Text]),
        "mstx": (bool, False),
        "mstx_domain_include": (list, []),
        "mstx_domain_exclude": (list, []),
        "schedule": (Schedule, None),
        "on_trace_ready": (Optional[Callable[..., Any]], None)
    }

    TYPE_INDEX = 0
    VALUE_INDEX = 1

    def __init__(self, **kwargs):
        self.is_set_schedule: bool = False
        self._set_schedule(**kwargs)
        self._check_deprecated_params(**kwargs)
        self._init_params(kwargs)
        self._check_params_type()
        self._handle_compatibility()

    @property
    def original_params(self) -> Dict[str, str]:
        """
        Get params dict for profiler_info.json save.
        """
        params = {}
        params["is_set_schedule"] = self.is_set_schedule
        for param, (_, _) in self.PARAMS.items():
            if param in ["profiler_level", "aic_metrics"]:
                params[param] = getattr(self, param).value
            elif param in ["activities", "export_type", "host_sys"]:
                params[param] = [item.value for item in getattr(self, param)]
            elif param == "schedule":
                params[param] = getattr(self, param).to_dict()
            elif param == "on_trace_ready":
                continue
            else:
                params[param] = getattr(self, param)
        return params

    @property
    def npu_profiler_params(self) -> Dict[str, str]:
        """
        Get NPU profiler parameters for Ascend profiler cpp backend.

        Returns:
            Dict[str, str]: A dictionary of NPU profiler parameters.
        """
        return {
            "profile_memory": self.profile_memory,
            "aicore_metrics": self.aic_metrics.value,
            "l2_cache": self.l2_cache,
            "hbm_ddr": self.hbm_ddr,
            "pcie": self.pcie,
            "sys_io": self.sys_io,
            "sys_interconnection": self.sys_interconnection,
            "host_sys": ",".join([item.value for item in self.host_sys]) if self.host_sys else "",
            "parallel_strategy": self.parallel_strategy,
            "profiler_level": self.profiler_level.value,
            "with_stack": self.with_stack,
            "record_shapes": self.record_shapes,
            "mstx": self.mstx,
            "mstx_domain_include": self.mstx_domain_include,
            "mstx_domain_exclude": self.mstx_domain_exclude,
            "cpu_trace": ProfilerActivity.CPU in self.activities,
            "npu_trace": ProfilerActivity.NPU in self.activities,
        }

    def _init_params(self, kwargs):
        """
        Initialize parameters with kwargs
        """
        for param, (_, default_value) in self.PARAMS.items():
            if param == "schedule" and kwargs.get(param) is None:
                kwargs["schedule"] = Schedule(wait=0, active=1)
            setattr(self, param, kwargs.get(param) if kwargs.get(param) is not None else default_value)

    def _check_params_type(self) -> None:
        """
        Check profiler input params type, if type is invalid reset to default value.
        """
        for key, value in self.__dict__.items():
            if key in ProfilerParameters.PARAMS:
                expected_type = ProfilerParameters.PARAMS[key][ProfilerParameters.TYPE_INDEX]
                default_value = ProfilerParameters.PARAMS[key][ProfilerParameters.VALUE_INDEX]
                if key == "on_trace_ready":
                    setattr(self, key, self._check_and_get_on_trace_ready(value, default_value))
                elif key == "schedule":
                    setattr(self, key, self._check_and_get_schedule(value))
                elif key == "export_type":
                    setattr(self, key, self._check_and_get_export_type(value))
                elif key in ("mstx_domain_include", "mstx_domain_exclude"):
                    setattr(self, key, self._check_and_get_mstx_domain(key, value))
                elif key == "host_sys":
                    setattr(self, key, self._check_and_get_host_sys(value, expected_type, default_value))
                # 检查可迭代类型
                elif isinstance(expected_type, type) and issubclass(expected_type, (list, tuple, set)):
                    setattr(self, key, self._check_and_get_iterable_params(key, value, expected_type, default_value))
                # 检查普通类型
                else:
                    setattr(self, key, self._check_and_get_common_params(key, value, expected_type, default_value))

    @staticmethod
    def _check_and_get_on_trace_ready(value, default_value):
        if value is not None and not callable(value):
            logger.warning(f"For Profiler, on_trace_ready value is Invalid, reset to {default_value}.")
            return default_value

        return value

    @staticmethod
    def _check_and_get_schedule(value):
        if not isinstance(value, Schedule):
            logger.warning(f"For Profiler, schedule value is Invalid, reset to {Schedule(wait=0, active=1)}")
            return Schedule(wait=0, active=1)

        return value

    @staticmethod
    def _check_and_get_iterable_params(key, value, expected_type, default_value):
        if not (isinstance(value, expected_type) and all(isinstance(item, type(default_value[0])) for item in value)):
            logger.warning(f"For Profiler, {key} value is Invalid, reset to {default_value}.")
            return default_value

        return list(set(value))

    @staticmethod
    def _check_and_get_common_params(key, value, expected_type, default_value):
        if not isinstance(value, expected_type):
            logger.warning(f"For Profiler, the type of {key} should be {expected_type}, "
                           f"but got {type(value)}, reset to {default_value}.")
            return default_value

        return value

    @staticmethod
    def _check_deprecated_params(**kwargs) -> None:
        """
        Check deprecated parameters.
        """
        for key, value in kwargs.items():
            if key == "profile_communication":
                warnings.warn(
                    "The parameter 'profile_communication' is deprecated,"
                    " please use 'profiler_level=ProfilerLevel.Level1' or "
                    "'profiler_level=ProfilerLevel.Level2' instead."
                )
            elif key == "op_time":
                warnings.warn(
                    "The parameter 'op_time' is deprecated,"
                    " please use 'activaties=ProfilerActivity.NPU' instead."
                )
            elif key == "profile_framework":
                warnings.warn(
                    "The parameter 'profile_framework' is deprecated,"
                    " please use 'activaties=ProfilerActivity.CPU' instead."
                )
            elif key == "host_stack":
                warnings.warn(
                    "The parameter 'host_stack' is deprecated,"
                    " please use 'with_stack' instead."
                )
            elif key == "timeline_limit":
                warnings.warn(
                    "The parameter 'timeline_limit' is deprecated and will have no effect"
                )
            elif key == "pcie" and value is True:
                warnings.warn(
                    "The parameter 'pcie' will be deprecated in future versions. "
                    "Please use 'sys_interconnection' in mindspore.profiler._ExperimentalConfig instead."
                )
            elif key == "hbm_ddr" and value is True:
                warnings.warn(
                    "The parameter 'hbm_ddr' will be deprecated in future versions. "
                    "Please use 'profile_memory' in instead."
                )

    def _set_schedule(self, **kwargs):
        if "schedule" in kwargs and isinstance(kwargs["schedule"], Schedule):
            self.is_set_schedule = True

    def _handle_compatibility(self) -> None:
        """
        Handle compatibility.
        """
        if hasattr(self, "schedule") and self.is_set_schedule and self.__dict__.get('data_process', False):
            self.data_process = False
            warnings.warn("When 'schedule' is set, 'data_process' will be set to False.")

        if not self.__dict__.get('mstx') and self.__dict__.get('profiler_level') == ProfilerLevel.LevelNone:
            self.profiler_level = ProfilerLevel.Level0
            warnings.warn("when 'mstx' is disabled, 'profiler_level' cannot be set to 'ProfilerLevel.LevelNone', "
                          "reset to 'ProfilerLevel.Level0'.")

        if self.__dict__.get('profiler_level') in (ProfilerLevel.LevelNone, ProfilerLevel.Level0) and \
            self.__dict__.get('aic_metrics') != AicoreMetrics.AiCoreNone:
            self.aic_metrics = AicoreMetrics.AiCoreNone
            warnings.warn(f"when 'profiler_level' is set to '{self.__dict__.get('profiler_level')}', "
                          f"'aic_metrics' cannot be set to other value except 'AicoreMetrics.AiCoreNone', "
                          f"reset to 'AicoreMetrics.AiCoreNone'.")

        if self.__dict__.get('profiler_level') in (ProfilerLevel.Level1, ProfilerLevel.Level2) and \
            self.__dict__.get('aic_metrics') == AicoreMetrics.AiCoreNone:
            self.aic_metrics = AicoreMetrics.PipeUtilization
            warnings.warn(f"when 'profiler_level' is set to '{self.__dict__.get('profiler_level')}', "
                          f"'aic_metrics' cannot be set to 'AicoreMetrics.AiCoreNone', "
                          f"reset to 'AicoreMetrics.PipeUtilization'.")

        if self.__dict__.get('record_shapes', False) and ProfilerActivity.CPU not in self.__dict__.get('activities'):
            self.record_shapes = False
            warnings.warn("when 'ProfilerActivity.CPU' is not set in 'activities', 'Record_shapes' cannot be set to "
                          "True, reset to 'False'.")

        if not self.__dict__.get('mstx') and (self.__dict__.get('mstx_domain_include') or \
            self.__dict__.get('mstx_domain_exclude')):
            self.mstx_domain_include = []
            self.mstx_domain_exclude = []
            warnings.warn(
                "when 'mstx' is set to 'False', 'mstx_domain_include' and 'mstx_domain_exclude' cannot be set, "
                "reset them to '[]'."
            )

        if self.__dict__.get('mstx_domain_include') and self.__dict__.get('mstx_domain_exclude'):
            self.mstx_domain_exclude = []
            warnings.warn(
                f"mstx_domain_include and mstx_domain_exclude can not be set together, "
                f"mstx_domain_exclude has been reset to {self.mstx_domain_exclude}."
            )

    @staticmethod
    def _check_and_get_export_type(export_type) -> list:
        """
        Check export type.
        """
        if not export_type:
            return [ExportType.Text]

        if isinstance(export_type, str):
            if export_type in (ExportType.Text.value, ExportType.Db.value):
                return [ExportType(export_type)]

        if isinstance(export_type, list):
            if all(isinstance(type, ExportType) for type in export_type):
                return list(set(export_type))

        if isinstance(export_type, ExportType):
            return [export_type]

        logger.warning("Invalid parameter export_type, reset it to text.")
        return [ExportType.Text]

    @staticmethod
    def _check_and_get_mstx_domain(list_name, domain_list) -> list:
        """
        Check mstx domain.
        """
        if not domain_list:
            return []
        if not isinstance(domain_list, list):
            logger.warning(f"For Profiler, {list_name} value is Invalid, reset to [].")
            return []
        for domain_name in domain_list:
            if not isinstance(domain_name, str) or domain_name == "":
                logger.warning(f"{list_name} has value {domain_name} is not str or is empty, reset to [].")
                return []
        return list(set(domain_list))

    @staticmethod
    def _check_and_get_host_sys(host_sys, expected_type, default_value):
        """
        Check host system.
        """
        if not host_sys:
            return default_value

        if not (isinstance(host_sys, expected_type) and
                all(isinstance(item, HostSystem) for item in host_sys)):
            logger.warning(
                f"For Profiler, 'host_sys' value is Invalid, reset to {default_value}."
            )
            return default_value

        return set(host_sys)

    def __getattr__(self, name):
        """
        Get attribute.
        """
        if name in self.PARAMS:
            return getattr(self, name)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
