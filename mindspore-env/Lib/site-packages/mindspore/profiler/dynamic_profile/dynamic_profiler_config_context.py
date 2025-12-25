# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Dynamic Profile config context"""
import json

from mindspore import log as logger
from mindspore.profiler.common.constant import (
    ProfilerActivity,
    ProfilerLevel,
    AicoreMetrics,
    ExportType,
    HostSystem
)
from mindspore.profiler.dynamic_profile.dynamic_profiler_utils import DynamicProfilerUtils


class DynamicProfilerConfigContext:
    """
    Data class for dynamic profile config context.
    """
    BOOL_MAP = {'true': True, 'false': False}

    def __init__(self, json_data):
        self._start_step = -1
        self._stop_step = -1
        self._aic_metrics = "AiCoreNone"
        self._profiler_level = "Level0"
        self._analyse_mode = -1
        self._activities = ["CPU", "NPU"]
        self._export_type = ["text"]
        self._profile_memory = False
        self._mstx = False
        self._parallel_strategy = False
        self._with_stack = False
        self._data_simplification = True
        self._l2_cache = False
        self._analyse = True
        self._is_valid = False
        self._record_shapes = False
        self._prof_path = "./"
        self._mstx_domain_include = []
        self._mstx_domain_exclude = []
        self._host_sys = []
        self._sys_io = False
        self._sys_interconnection = False
        self._is_dyno = DynamicProfilerUtils.is_dyno_mode()
        self._parse(json_data)
        self._check_params_type()
        self._json_dict = self.to_dict()

    def _parse(self, json_data):
        """ Parse the given JSON data and initialize the attributes of the class based on its content."""
        self._parse_start_step(json_data)
        self._parse_stop_step(json_data)
        self._parse_profiler_level(json_data)
        self._parse_activities(json_data)
        self._parse_export_type(json_data)
        self._parse_profiler_memory(json_data)
        self._parse_mstx(json_data)
        self._parse_with_stack(json_data)
        self._parse_data_simplification(json_data)
        self._parse_l2_cache(json_data)
        self._parse_analyse(json_data)
        self._parse_record_shapes(json_data)
        self._parse_prof_path(json_data)
        self._aic_metrics = json_data.get("aic_metrics", "AiCoreNone")
        self._analyse_mode = json_data.get("analyse_mode", -1)
        self._parse_mstx_domain_include(json_data)
        self._parse_mstx_domain_exclude(json_data)
        self._parse_host_sys(json_data)
        self._parse_sys_io(json_data)
        self._parse_sys_interconnection(json_data)
        self._parallel_strategy = json_data.get("parallel_strategy", False)
        self._is_valid = json_data.get("is_valid", False)

    def _parse_start_step(self, json_data):
        """ Parse the start_step from JSON data."""
        if self._is_dyno:
            try:
                start_step = int(json_data.get("start_step", 0))
            except ValueError:
                start_step = 0
                logger.warning("dyno config 'start-step' should be an integer, will be reset to default value: '0'.")
            self._start_step = start_step
        else:
            self._start_step = json_data.get("start_step", -1)

    def _parse_stop_step(self, json_data):
        """ Parse the stop_step from JSON data."""
        if self._is_dyno:
            try:
                start_step = int(json_data.get("start_step", 0))
            except ValueError:
                start_step = 0
                logger.warning("dyno config 'start-step' should be an integer, will be reset to default value: '0'.")

            try:
                iterations = int(json_data.get("iterations", 0))
            except ValueError:
                iterations = 0
                logger.warning("dyno config 'iterations' should be an integer, will be reset to default value: '0'.")

            self._stop_step = start_step + iterations - 1
        else:
            self._stop_step = json_data.get("stop_step", -1)

    def _parse_profiler_level(self, json_data):
        """ Parse the profiler_level from JSON data."""
        if self._is_dyno:
            profiler_level = json_data.get("profiler_level", "Level0")
            if profiler_level == "Level_none":
                profiler_level = "LevelNone"
            self._profiler_level = profiler_level
        else:
            self._profiler_level = json_data.get("profiler_level", "Level0")

    def _parse_activities(self, json_data):
        """ Parse the activities from JSON data."""
        if self._is_dyno:
            activities = json_data.get("activities", "CPU, NPU")
            activity_map = {
                "CPU, NPU": ["CPU", "NPU"],
                "NPU, CPU": ["CPU", "NPU"],
                "CPU": ["CPU"],
                "NPU": ["NPU"]
            }
            self._activities = activity_map.get(activities, ["CPU", "NPU"])
        else:
            self._activities = json_data.get("activities", ["CPU", "NPU"])

    def _parse_export_type(self, json_data):
        """ Parse the export_type from JSON data."""
        if self._is_dyno:
            export_type = json_data.get("export_type", "Text")
            export_types_map = {
                "Text": ["text"],
                "Db": ["db"],
            }
            self._export_type = export_types_map.get(export_type, ["text"])
        else:
            self._export_type = json_data.get("export_type", ["text"])

    def _parse_profiler_memory(self, json_data):
        """ Parse the profiler_memory from JSON data."""
        if self._is_dyno:
            profile_memory = json_data.get("profile_memory", "")
            self._profile_memory = self.BOOL_MAP.get(profile_memory.lower(), False)
        else:
            self._profile_memory = json_data.get("profile_memory", False)

    def _parse_mstx(self, json_data):
        """ Parse the mstx from JSON data."""
        if self._is_dyno:
            mstx = json_data.get("msprof_tx", "")
            self._mstx = self.BOOL_MAP.get(mstx.lower(), False)
        else:
            self._mstx = json_data.get("mstx", False)

    def _parse_with_stack(self, json_data):
        """ Parse the with_stack from JSON data."""
        if self._is_dyno:
            with_stack = json_data.get("with_stack", "")
            self._with_stack = self.BOOL_MAP.get(with_stack.lower(), False)
        else:
            self._with_stack = json_data.get("with_stack", False)

    def _parse_data_simplification(self, json_data):
        """ Parse the data_simplification from JSON data."""
        if self._is_dyno:
            data_simplification = json_data.get("data_simplification", "")
            self._data_simplification = self.BOOL_MAP.get(data_simplification.lower(), False)
        else:
            self._data_simplification = json_data.get("data_simplification", True)

    def _parse_l2_cache(self, json_data):
        """ Parse the l2_cach from JSON data."""
        if self._is_dyno:
            l2_cache = json_data.get("l2_cache", "")
            self._l2_cache = self.BOOL_MAP.get(l2_cache.lower(), False)
        else:
            self._l2_cache = json_data.get("l2_cache", False)

    def _parse_analyse(self, json_data):
        """ Parse the data_simplification from JSON data."""
        if self._is_dyno:
            analyse = json_data.get("analyse", "")
            self._analyse = self.BOOL_MAP.get(analyse.lower(), False)
        else:
            self._analyse = json_data.get("analyse", False)

    def _parse_record_shapes(self, json_data):
        """ Parse the record_shapes from JSON data."""
        if self._is_dyno:
            record_shapes = json_data.get("record_shapes", "")
            self._record_shapes = self.BOOL_MAP.get(record_shapes.lower(), False)
        else:
            self._record_shapes = json_data.get("record_shapes", False)

    def _parse_prof_path(self, json_data):
        """ Parse the prof_path from JSON data."""
        if self._is_dyno:
            prof_path = json_data.get("log_file", "./")
            if not isinstance(prof_path, str):
                logger.warning("The 'log-file' must be a string, "
                               "will be set to default: './'.")
                prof_path = "./"
            self._prof_path = prof_path
        else:
            self._prof_path = json_data.get("prof_path", "./")

    def _parse_host_sys(self, json_data):
        """ Parse the host_sys from JSON data."""
        if self._is_dyno:
            host_sys = json_data.get("host_sys", None)
            self._host_sys = [] if host_sys is None or host_sys == "None" else \
                [item.strip() for item in host_sys.split(',')]
        else:
            self._host_sys = json_data.get("host_sys", [])

    def _parse_sys_io(self, json_data):
        """ Parse the sys_io from JSON data."""
        if self._is_dyno:
            sys_io = json_data.get("sys_io", False)
            if isinstance(sys_io, str):
                self._sys_io = self.BOOL_MAP.get(sys_io.lower(), False)
            else:
                self._sys_io = False
        else:
            self._sys_io = json_data.get("sys_io", False)

    def _parse_sys_interconnection(self, json_data):
        """ Parse the sys_interconnection from JSON data."""
        if self._is_dyno:
            sys_interconnection = json_data.get("sys_interconnection", False)
            if isinstance(sys_interconnection, str):
                self._sys_interconnection = self.BOOL_MAP.get(sys_interconnection.lower(), False)
            else:
                self._sys_interconnection = False
        else:
            self._sys_interconnection = json_data.get("sys_interconnection", False)

    def _parse_mstx_domain_include(self, json_data):
        """ Parse the mstx_domain_include from JSON data."""
        if self._is_dyno:
            mstx_domain_include = json_data.get("mstx_domain_include", None)
            self._mstx_domain_include = [] if mstx_domain_include is None or mstx_domain_include == "None" else \
                [item.strip() for item in mstx_domain_include.split(',')]
        else:
            self._mstx_domain_include = json_data.get("mstx_domain_include", [])

    def _parse_mstx_domain_exclude(self, json_data):
        """ Parse the mstx_domain_exclude from JSON data."""
        if self._is_dyno:
            mstx_domain_exclude = json_data.get("mstx_domain_exclude", None)
            self._mstx_domain_exclude = [] if mstx_domain_exclude is None or mstx_domain_exclude == "None" else \
                [item.strip() for item in mstx_domain_exclude.split(',')]
        else:
            self._mstx_domain_exclude = json_data.get("mstx_domain_exclude", [])

    def _check_params_type(self):
        """ Check and enforce parameter types with lower complexity."""
        # Check non-special parameters. {Parameter name: (expected type, default value)}
        self._check_non_special_params()
        # Check special parameters
        self._check_special_params()

    def _check_non_special_params(self):
        """ Check non-special parameters."""
        param_rules = {
            '_start_step': (int, -1),
            '_stop_step': (int, -1),
            '_analyse_mode': (int, -1),
            '_profile_memory': (bool, False),
            '_mstx': (bool, False),
            '_l2_cache': (bool, False),
            '_analyse': (bool, False),
            '_parallel_strategy': (bool, False),
            '_with_stack': (bool, False),
            '_data_simplification': (bool, True),
            '_record_shapes': (bool, False),
            '_mstx_domain_include': (list, []),
            '_mstx_domain_exclude': (list, []),
            '_host_sys': (list, []),
            '_sys_io': (bool, False),
            '_sys_interconnection': (bool, False)
        }

        def _is_valid_type(value, expected_type):
            """ Helper method for type checking."""
            if expected_type is int and isinstance(value, bool):
                return False
            return isinstance(value, expected_type)

        for param, (expected_type, default) in param_rules.items():
            value = getattr(self, param)
            if not _is_valid_type(value, expected_type):
                logger.warning(
                    f"'{param[1:]}' should be {expected_type.__name__}, "
                    f"but got {type(value).__name__}. "
                    f"will be reset to default value: '{default}'."
                )
                setattr(self, param, default)

    def _check_special_params(self):
        """ Check special parameters."""
        self._check_aic_metrics()
        self._check_profiler_level()
        self._check_activities()
        self._check_export_type()
        self._check_prof_path()

    def _check_aic_metrics(self):
        """ Check aic_metrics."""
        if not (isinstance(self._aic_metrics, str) or (
                isinstance(self._aic_metrics, int) and not isinstance(self._aic_metrics, bool))):
            logger.warning(
                f"'aic_metrics' should be a string or a non-bool integer, "
                f"but got {type(self._aic_metrics).__name__}. "
                f"Will be reset to default value: 'AiCoreNone'."
            )
            self._aic_metrics = AicoreMetrics.AiCoreNone.value

    def _check_profiler_level(self):
        """ Check profiler_level."""
        if not (isinstance(self._profiler_level, str) or (
                isinstance(self._profiler_level, int) and not isinstance(self._profiler_level, bool))):
            logger.warning(
                f"'profiler_level' should be a string or a non-bool integer, "
                f"but got {type(self._profiler_level).__name__}. "
                f"Will be reset to default value: 'Level0'."
            )
            self._profiler_level = ProfilerLevel.Level0.value

    def _check_activities(self):
        """ Check activities."""
        if not (isinstance(self._activities, list) or (
                isinstance(self._activities, int) and not isinstance(self._activities, bool))):
            logger.warning(
                f"'activities' should be a list or a non-bool integer, "
                f"but got {type(self._activities).__name__}. "
                f"Will be reset to default value: '[ProfilerActivity.CPU, ProfilerActivity.NPU]'."
            )
            self._activities = [ProfilerActivity.CPU.value, ProfilerActivity.NPU.value]

    def _check_export_type(self):
        """ Check export_type."""
        if not (isinstance(self._export_type, list) or (
                isinstance(self._export_type, int) and not isinstance(self._export_type, bool))):
            logger.warning(
                f"'export_type' should be a list or a non-bool integer, "
                f"but got {type(self._export_type).__name__}. "
                f"Will be reset to default value: '[ExportType.Text]'."
            )
            self._export_type = [ExportType.Text.value]

    def _check_prof_path(self):
        """ Check prof_path."""
        if not isinstance(self._prof_path, str):
            logger.warning(f"'prof_path' should be str, but got {type(self._prof_path).__name__}. ")
            self._prof_path = "./"

    @property
    def start_step(self):
        """ Get start step value."""
        return self._start_step

    @property
    def stop_step(self):
        """ Get stop step value."""
        return self._stop_step

    @property
    def is_valid(self):
        """ Get json valid value."""
        return self._is_valid

    @is_valid.setter
    def is_valid(self, value):
        """ Set json valid value."""
        self._is_valid = value

    @property
    def analyse_mode(self):
        """ Get analyse mode value."""
        return self._convert_analyse_mode(self._analyse_mode)

    @property
    def prof_path(self):
        """ Get analyse mode value."""
        return self._prof_path

    @property
    def analyse(self):
        """ Get analyse mode value."""
        return self._analyse

    @property
    def vars(self):
        """ Get all values in DynamicProfilerConfigContext."""
        not_supported_args = ['_is_valid']
        res = {}
        for key, value in self.__dict__.items():
            if key in ['_json_dict', '_is_dyno']:
                continue
            if key not in not_supported_args:
                res[key.replace('_', '', 1)] = value
        return res

    @property
    def args(self):
        """ Get all args in DynamicProfilerConfigContext."""
        self._profiler_level = self._convert_profiler_level(self._profiler_level)
        self._activities = self._convert_activities(self._activities)
        self._aic_metrics = self._convert_aic_metrics(self._aic_metrics)
        self._export_type = self._convert_export_type(self._export_type)
        self._host_sys = self._convert_host_sys(self._host_sys)
        not_supported_args = ['_start_step', '_stop_step', '_analyse_mode', '_is_valid']
        res = {}
        for key, value in self.__dict__.items():
            if key not in not_supported_args:
                res[key.replace('_', '', 1)] = value
        return res

    def to_dict(self):
        """Convert the instance attributes to a dictionary."""
        return {
            "start_step": self._start_step,
            "stop_step": self._stop_step,
            "aic_metrics": self._aic_metrics,
            "profiler_level": self._profiler_level,
            "analyse_mode": self._analyse_mode,
            "activities": self._activities,
            "export_type": self._export_type,
            "profile_memory": self._profile_memory,
            "mstx": self._mstx,
            "parallel_strategy": self._parallel_strategy,
            "with_stack": self._with_stack,
            "data_simplification": self._data_simplification,
            "l2_cache": self._l2_cache,
            "analyse": self._analyse,
            "record_shapes": self._record_shapes,
            "prof_path": self._prof_path,
            "mstx_domain_include": self._mstx_domain_include,
            "mstx_domain_exclude": self._mstx_domain_exclude,
            "sys_io": self._sys_io,
            "sys_interconnection": self._sys_interconnection,
            "host_sys": self._host_sys,
            "is_valid": self._is_valid
        }

    def to_bytes(self) -> bytes:
        """ Convert dynamic profiler config context to a byte sequence."""
        return self.json_to_bytes(self._json_dict)

    @staticmethod
    def json_to_bytes(json_data):
        """ Convert a json to a byte sequence."""
        cfg_json_str = json.dumps(json_data)
        # Encode the JSON string as a byte sequence (using UTF-8 encoding)
        return cfg_json_str.encode("utf-8")

    @staticmethod
    def bytes_to_json(bytes_shm):
        """ Convert a byte sequence to a json."""
        try:
            json_string = bytes_shm.decode("utf-8")
            json_data = json.loads(json_string)
        except UnicodeDecodeError:
            logger.error("Failed to parse dynamic profiler config JSON string.")
            return {}

        return json_data

    @staticmethod
    def _convert_analyse_mode(analyse_mode: int):
        """Convert analyse_mode to real args in Profiler."""
        mode_map = {
            0: 'sync',
            1: 'async',
            -1: None
        }
        if analyse_mode in mode_map:
            return mode_map[analyse_mode]
        logger.warning(f"'analyse_mode' needs to be set to one of '-1, 0, and 1', but got '{analyse_mode}', "
                       f"will be reset to default: 'None'.")
        return None

    def _convert_profiler_level(self, profiler_level):
        """ Convert profiler_level to real args in Profiler."""
        # Convert int profiler_level
        if isinstance(profiler_level, int):
            logger.warning(f"The parameter 'profiler_level={profiler_level}' does not support passing in the "
                           f"int type in future versions. Please use the str type instead.")
            return self._convert_int_profiler_level(profiler_level)

        try:
            profiler_level = ProfilerLevel(profiler_level)
        except ValueError:
            logger.warning(
                f"'{profiler_level}' is not a valid profiler_level, "
                f"will be reset to will be reset to default: 'Level0'."
            )
            return ProfilerLevel.Level0

        return profiler_level

    @staticmethod
    def _convert_int_profiler_level(profiler_level):
        """ Convert int profiler_level to real args in Profiler."""
        if profiler_level == -1:
            return ProfilerLevel.LevelNone
        if profiler_level == 0:
            return ProfilerLevel.Level0
        if profiler_level == 1:
            return ProfilerLevel.Level1
        if profiler_level == 2:
            return ProfilerLevel.Level2

        logger.warning(f"'profiler_level' needs to be set to one of '-1, 0, 1 and 2', but got '{profiler_level}',"
                       f"will be reset to default: '{ProfilerLevel.Level0}'.")
        return ProfilerLevel.Level0

    def _convert_activities(self, activities):
        """ Convert activities to real args in Profiler."""
        # Convert int activities
        if isinstance(activities, int):
            logger.warning(f"The parameter 'activities={activities}' does not support passing in the int type in "
                           f"future versions. Please use the list type instead.")
            return self._convert_int_activities(activities)

        converted_activities = []

        for activity in activities:
            try:
                converted_activity = ProfilerActivity(activity)
                converted_activities.append(converted_activity)
            except ValueError:
                logger.warning(f"'{activity}' is not a valid ProfilerActivity member. "
                               f"will be reset to default: '{[ProfilerActivity.CPU, ProfilerActivity.NPU]}'.")
                return [ProfilerActivity.CPU, ProfilerActivity.NPU]

        return converted_activities

    @staticmethod
    def _convert_int_activities(activities):
        """ Convert int activities to real args in Profiler."""
        if activities == 0:
            return [ProfilerActivity.CPU, ProfilerActivity.NPU]
        if activities == 1:
            return [ProfilerActivity.CPU]
        if activities == 2:
            return [ProfilerActivity.NPU]

        logger.warning(f"'activities' needs to be set to one of '0, 1 and 2', but got '{activities}',"
                       f"will be reset to default: '{[ProfilerActivity.CPU, ProfilerActivity.NPU]}'.")
        return [ProfilerActivity.CPU, ProfilerActivity.NPU]

    def _convert_aic_metrics(self, aic_metrics):
        """ Convert aic_metrics to real args in Profiler."""
        # Convert int aic_metrics
        if isinstance(aic_metrics, int):
            logger.warning(f"The parameter 'aic_metrics={aic_metrics}' does not support passing in the int type in "
                           f"future versions. Please use the str type instead.")
            return self._convert_int_aic_metrics(aic_metrics)

        # Special handling of the AiCoreNone scene
        if aic_metrics == "AiCoreNone":
            return AicoreMetrics.AiCoreNone

        try:
            aic_metrics = AicoreMetrics(aic_metrics)
        except ValueError:
            logger.warning(
                f"'{aic_metrics}' is not a valid aic_metrics, "
                f"will be reset to will be reset to default: 'AiCoreNone'."
            )
            return AicoreMetrics.AiCoreNone

        return aic_metrics

    @staticmethod
    def _convert_int_aic_metrics(aic_metrics):
        """ Convert int aic_metrics to real args in Profiler."""
        if aic_metrics == 0:
            return AicoreMetrics.PipeUtilization
        if aic_metrics == 1:
            return AicoreMetrics.ArithmeticUtilization
        if aic_metrics == 2:
            return AicoreMetrics.Memory
        if aic_metrics == 3:
            return AicoreMetrics.MemoryL0
        if aic_metrics == 4:
            return AicoreMetrics.MemoryUB
        if aic_metrics == 5:
            return AicoreMetrics.ResourceConflictRatio
        if aic_metrics == 6:
            return AicoreMetrics.L2Cache
        if aic_metrics == 7:
            return AicoreMetrics.MemoryAccess

        logger.warning(f"'aic_metrics' needs to be set to one of '0, 1, 2, 3, 4, 5, 6, and 7',"
                       f"but got '{aic_metrics}', will be reset to default: '{AicoreMetrics.AiCoreNone}'.")
        return AicoreMetrics.AiCoreNone

    def _convert_export_type(self, export_types):
        """ Convert export_type to real args in Profiler."""
        # Convert int export_type
        if isinstance(export_types, int):
            logger.warning(f"The parameter 'export_type={export_types}' does not support passing in the int type "
                           f"in future versions. Please use the list type instead.")
            return self._convert_int_export_type(export_types)

        converted_export_types = []

        for export_type in export_types:
            try:
                converted_export_type = ExportType(export_type)
                converted_export_types.append(converted_export_type)
            except ValueError:
                logger.warning(f"'{export_type}' is not a valid ExportType member. "
                               f"will be reset to default: '{[ExportType.Text]}'.")
                return [ExportType.Text]

        return converted_export_types

    @staticmethod
    def _convert_int_export_type(export_types):
        """ Convert int export_type to real args in Profiler."""
        if export_types == 1:
            return [ExportType.Db]
        if export_types == 2:
            return [ExportType.Text, ExportType.Db]

        logger.warning(f"'export_types' needs to be set to one of '1 and 2', but got '{export_types}', "
                       f"will be reset to default: '{[ExportType.Text]}'.")
        return [ExportType.Text]

    @staticmethod
    def _convert_host_sys(host_systems):
        """ Convert host_sys to real args in Profiler."""
        if not host_systems:
            return None

        converted_host_systems = []
        for host_system in host_systems:
            try:
                converted_host_sys = HostSystem(host_system)
                converted_host_systems.append(converted_host_sys)
            except ValueError:
                logger.warning(f"'{host_system}' is not a valid HostSystem member. "
                               f"will be reset to default: 'None'.")
                return None

        return converted_host_systems
