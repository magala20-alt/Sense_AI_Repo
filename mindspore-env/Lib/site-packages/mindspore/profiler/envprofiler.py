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
"""EnvProfiler"""
import os
import json
from mindspore import log as logger
from mindspore.profiler.profiler import Profile
from mindspore.profiler.experimental_config import _ExperimentalConfig
from mindspore.profiler import tensorboard_trace_handler
from mindspore.profiler.common.constant import (
    ProfilerLevel,
    AicoreMetrics,
    ProfilerActivity,
    ExportType,
    HostSystem
)
from mindspore.profiler.common.profiler_parameters import ProfilerParameters


class EnvProfiler:
    """Collect and analyze training performance data, support calls during and after training."""

    NOT_SUPPORTED_PARAMS = ["schedule", "on_trace_ready"]
    profiler = None

    @classmethod
    def init_profiler(cls):
        """
        Initialize the profiler.
        """
        if not os.getenv("MS_PROFILER_OPTIONS"):
            return
        options = cls._load_options()
        if not options:
            logger.error("Failed to load MS_PROFILER_OPTIONS, json decode error.")
            return

        params = cls._convert_options_to_profiler_params(options)
        logger.info(f"params: {params}")
        if params["start"]:
            experimental_config = _ExperimentalConfig(profiler_level=params.get("profiler_level"),
                                                      aic_metrics=params.get("aic_metrics"),
                                                      l2_cache=params.get("l2_cache"),
                                                      mstx=params.get("mstx"),
                                                      data_simplification=params.get("data_simplification"),
                                                      export_type=params.get("export_type"),
                                                      mstx_domain_include=params.get("mstx_domain_include"),
                                                      mstx_domain_exclude=params.get("mstx_domain_exclude"),
                                                      sys_io=params.get("sys_io"),
                                                      sys_interconnection=params.get("sys_interconnection"),
                                                      host_sys=params.get("host_sys"))
            cls.profiler = Profile(activities=params.get("activities"),
                                   with_stack=params.get("with_stack"),
                                   profile_memory=params.get("profile_memory"),
                                   data_process=params.get("data_process"),
                                   parallel_strategy=params.get("parallel_strategy"),
                                   start_profile=params.get("start_profile"),
                                   hbm_ddr=params.get("hbm_ddr"),
                                   pcie=params.get("pcie"),
                                   sync_enable=params.get("sync_enable"),
                                   record_shapes=params.get("record_shapes"),
                                   on_trace_ready=tensorboard_trace_handler(params.get("output_path")),
                                   experimental_config=experimental_config)
            cls.profiler.start()
            logger.info("Profiler init success.")

    def analyse(self):
        """
        Analyze the collected data.
        """
        logger.info("analyse start")
        if not self.profiler:
            logger.info("Profiler is not initialized, skip analyse.")
            return
        self.profiler.stop()
        logger.info("analyse end")

    @classmethod
    def _load_options(cls):
        """
        Load the options from the environment variable.
        """
        try:
            options = json.loads(os.environ.get("MS_PROFILER_OPTIONS", "{}"))
        except json.JSONDecodeError:
            return {}
        return options

    @classmethod
    def _convert_option_to_enum_value(cls, enum_class, option_value, default_value):
        """
        Convert the option value to the enum value.
        """
        try:
            return enum_class(option_value)
        except ValueError:
            logger.warning(
                f"The value '{option_value}' of parameter '{enum_class.__name__}' is invalid, "
                f"use default value '{default_value}' instead."
            )
            return default_value

    @classmethod
    def _convert_options_to_profiler_params(cls, options):
        """
        Convert the options to the profiler parameters.
        """
        params = {}
        if not options:
            logger.warning("MS_PROFILER_OPTIONS is empty, use default values.")
            return params

        if "output_path" in options:
            params["output_path"] = options["output_path"]

        # if start is not set, default is False
        params["start"] = options.get("start", False)

        for param, (_, default_value) in ProfilerParameters.PARAMS.items():
            if param in options and param not in cls.NOT_SUPPORTED_PARAMS:
                if param == "activities" and isinstance(options[param], list):
                    params[param] = cls._convert_enums_to_list(
                        options[param], default_value, ProfilerActivity
                    )
                elif param == "host_sys" and isinstance(options[param], list):
                    params[param] = cls._convert_enums_to_list(
                        options[param], default_value, HostSystem
                    )
                elif param == "aic_metrics":
                    params[param] = cls._convert_option_to_enum_value(
                        AicoreMetrics, options[param], default_value
                    )
                elif param == "profiler_level":
                    params[param] = cls._convert_option_to_enum_value(
                        ProfilerLevel, options[param], default_value
                    )
                elif param == "export_type":
                    params[param] = cls._convert_export_type_to_list(
                        options[param], default_value
                    )
                else:
                    params[param] = options[param]
        return params

    @classmethod
    def _convert_enums_to_list(cls, values, default_value, enum_class):
        """
        Convert the enums to the list.
        """
        res = []
        for value in values:
            res.append(
                cls._convert_option_to_enum_value(
                    enum_class, value, default_value
                )
            )
        # remove duplicate
        return list(set(default_value if default_value in res else res))

    @classmethod
    def _convert_export_type_to_list(cls, export_types, default_value) -> list:
        """
        Check the export type to the list.
        """
        res = []
        for export_type in export_types:
            if export_type not in ("text", "db"):
                logger.warning(
                    f"The value '{export_type}' of parameter '{ExportType.__name__}' is invalid, "
                    f"use default value '{default_value}' instead."
                )
                return default_value
            res.append(
                cls._convert_option_to_enum_value(
                    ExportType, export_type, default_value
                )
            )
        # remove duplicate
        return list(set(res))

EnvProfiler.init_profiler()
