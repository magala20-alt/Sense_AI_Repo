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
"""mind data viewer"""
import os
from queue import Queue
from collections import defaultdict
from typing import List, Dict, Any

from mindspore import log as logger
from mindspore.profiler.common.constant import ProfilerActivity
from mindspore.profiler.analysis.viewer.base_viewer import BaseViewer
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.log import ProfilerLogger
from mindspore.profiler.common.exceptions.exceptions import (
    ProfilerRawFileException,
)


class MindDataPipelineRawViewer(BaseViewer):
    """
    MindData Pipeline Raw Viewer for parsing and saving raw pipeline profiling data.
    This class processes the raw pipeline information and saves it in a CSV format.
    """
    _FILE_NAME = 'minddata_pipeline_raw_{}.csv'
    _COL_NAMES = [
        'op_id', 'op_type', 'num_workers', 'output_queue_size',
        'output_queue_average_size', 'output_queue_length',
        'output_queue_usage_rate', 'sample_interval', 'parent_id', 'children_id'
    ]

    def __init__(self, **kwargs):
        super().__init__()
        self._device_id = kwargs.get("rank_id") if (ProfilerActivity.NPU.value in
                                                    kwargs.get("activities")) else kwargs.get("device_id")
        self._save_path = os.path.join(
            kwargs.get("ascend_profiler_output_path"),
            self._FILE_NAME.format(self._device_id)
        )
        self._ascend_ms_dir = kwargs.get("ascend_ms_dir")
        ProfilerLogger.init(self._ascend_ms_dir)
        self._logger = ProfilerLogger.get_instance()

    def save(self, data: Dict[str, Any]) -> None:
        if not data.get("pipeline_info"):
            return
        try:
            dict_op_id_info, sampling_interval = data["pipeline_info"]
            op_info_list = self._analyse_data(dict_op_id_info, sampling_interval)
            self._logger.info("Analyse minddata pipeline raw data done")
            self._save_data(op_info_list)
            self._logger.info("Save minddata pipeline raw data done")
        except Exception as e: # pylint: disable=W0703
            self._logger.error("Failed to save minddata %s", e, exc_info=True)

    def _analyse_data(self, dict_op_id_info: Dict[int, Dict[str, Any]], sample_interval: float) -> List[List[Any]]:
        """
        Create a list of operator information from the op id info dict.
        Args:
            dict_op_id_info (dict): dict of operator information indexed by op_id.
            sample_interval (int): Sampling interval for the profiling data.
        Returns:
            list: List of operator information for each node in the pipeline.
        Raises:
            ProfilerRawFileException: If the root operator (id=0) does not exist.
        """
        root_node = dict_op_id_info.get(0)
        if not root_node:
            raise ProfilerRawFileException(
                'The format of minddata pipeline raw file is wrong, '
                'the operator that id is 0 does not exist.'
            )
        root_node['parent_id'] = None
        queue = Queue()
        queue.put_nowait(root_node)
        op_info_list = []

        while not queue.empty():
            node = queue.get_nowait()
            self._update_child_node(node, dict_op_id_info)
            op_info_list.append(self._get_op_info(node, sample_interval))
            for child_op_id in node.get('children', []):
                sub_node = dict_op_id_info.get(child_op_id)
                sub_node['parent_id'] = node['op_id']
                queue.put_nowait(sub_node)

        return op_info_list

    def _save_data(self, op_info_list: List[List[Any]]) -> None:
        self._logger.info("Save minddata pipeline raw data start")
        FileManager.create_csv_file(self._save_path, op_info_list, self._COL_NAMES)
        self._logger.info(
            "Save minddata pipeline raw data done, %d rows saved, save path: %s",
            len(op_info_list),
            self._save_path,
        )

    @staticmethod
    def _update_child_node(node: Dict[str, Any], dict_op_id_info: Dict[int, Dict[str, Any]]) -> None:
        """
        Update the children information for a given node in the pipeline.
        Args:
            node (dict): The current node (operator) being processed.
            dict_op_id_info (dict): Dict of operator information indexed by op_id.
        """
        child_op_ids = node.get('children', [])
        if not child_op_ids:
            return

        queue = Queue()
        queue.queue.extend(child_op_ids)

        new_child_op_ids = []
        while not queue.empty():
            child_op_id = queue.get_nowait()
            child_node = dict_op_id_info.get(child_op_id)
            if not child_node:
                continue
            metrics = child_node.get('metrics')
            if not metrics or not metrics.get('output_queue'):
                children = child_node.get('children', [])
                if children:
                    queue.queue.extend(children)
            else:
                new_child_op_ids.append(child_op_id)

        node['children'] = new_child_op_ids

    @staticmethod
    def _get_op_info(op_node: Dict[str, Any], sample_interval: float) -> List[Any]:
        """
        Get the operator information.
        Args:
            op_node (dict):  The node represents an operator.
            sample_interval (int): Sampling interval.
        Returns:
            list: A list containing the extracted operator information.
        Raises:
            ValueError: If the queue size is None or the queue length is 0.
        """
        metrics = op_node.get('metrics')
        output_queue_info = metrics.get('output_queue', {}) if metrics else {}
        queue_size = output_queue_info.get('size')
        queue_length = output_queue_info.get('length')

        if output_queue_info and queue_size is None:
            raise ValueError("The queue size cannot be None.")
        if output_queue_info and queue_length == 0:
            raise ValueError("The length of queue cannot be 0.")

        queue_average_size = sum(queue_size) / len(queue_size) if queue_size else None
        queue_usage_rate = queue_average_size / queue_length if queue_average_size is not None else None

        children_id = op_node.get('children')
        return [
            op_node.get('op_id'),
            op_node.get('op_type'),
            op_node.get('num_workers'),
            queue_size,
            queue_average_size,
            queue_length,
            queue_usage_rate,
            sample_interval,
            op_node.get('parent_id'),
            children_id if children_id else None
        ]


class MindDataPiplineSummaryViewer(BaseViewer):
    """
    MindData Pipeline Summary Viewer for processing and saving pipeline profiling summary data.
    This class calculates various metrics and saves the summary in both JSON and CSV formats.
    """
    _FILE_NAMES = {
        'json_file': 'minddata_pipeline_summary_{}.json',
        'csv_file': 'minddata_pipeline_summary_{}.csv'
    }

    def __init__(self, **kwargs):
        super().__init__()
        self._device_id = kwargs.get("rank_id") if (ProfilerActivity.NPU.value in
                                                    kwargs.get("activities")) else kwargs.get("device_id")
        self._device_queue_file_found = None
        self._save_paths = {
            file_type: os.path.join(kwargs.get("ascend_profiler_output_path"), file_name.format(self._device_id))
            for file_type, file_name in self._FILE_NAMES.items()
        }
        self._ascend_ms_dir = kwargs.get("ascend_ms_dir")
        ProfilerLogger.init(self._ascend_ms_dir, "MindDataPiplineSummaryViewer")
        self._logger = ProfilerLogger.get_instance()

    def save(self, data: Dict[str, Any]) -> None:
        # If there are errors in the data during the parsing phase, the data will be set to empty
        if not (data.get("pipeline_info") and data.get('cpu_util_info') and data.get("device_trace_info")):
            return
        try:
            self._device_queue_file_found = data["device_queue_file_found"]
            summary_dict = self._analyse_data(data)
            self._logger.info("Analyse minddata pipeline summary data done")
            self._save_data(summary_dict)
            self._logger.info("Save minddata pipeline summary data done")
        except Exception as e: # pylint: disable=W0703
            self._logger.error("Failed to save minddata %s", e, exc_info=True)

    def _analyse_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the summary information from the raw profiling data.
        Args:
            data (dict): Raw profiling data.
        Returns:
            dict: Calculated summary information.
        """
        summary_dict = {}
        summary_dict.update(self._analyse_pipeline_info(data["pipeline_info"]))
        summary_dict.update(self._analyse_cpu_util_info(data["cpu_util_info"]))
        summary_dict.update(self._analyse_device_trace_info(data["device_trace_info"]))
        self._check_and_update_summary(summary_dict)
        return summary_dict

    def _save_data(self, summary_dict: Dict[str, Any]) -> None:
        self._logger.info("Save minddata pipeline summary data start")
        FileManager.create_json_file(self._save_paths['json_file'], summary_dict)
        FileManager.create_csv_file(
            self._save_paths['csv_file'],
            [[key] + value for key, value in summary_dict.items()]
        )
        self._logger.info("Save minddata pipeline summary data done")

    def _check_and_update_summary(self, summary_dict: Dict[str, Any]) -> None:
        """
        Check for consistency in the summary data and update with composite information.
        Args:
            summary_dict (dict): Summary dictionary to be checked and updated.
        """
        num_pipeline_ops = len(summary_dict.get('pipeline_ops', []))
        num_cpu_util_ops = len(summary_dict.get('avg_cpu_pct', []))
        # Check if both pipeline data and CPU utilization data have the same number of ops
        if num_pipeline_ops == num_cpu_util_ops:
            summary_dict.update(self._analyse_composite_info(summary_dict))
            bottleneck_dict = self._analyse_bottleneck_op(summary_dict)
            if bottleneck_dict:
                summary_dict.update(bottleneck_dict)
        else:
            logger.warning(f'Number of ops mismatch: pipeline data ({num_pipeline_ops}) '
                           f'vs CPU utilization data ({num_cpu_util_ops})')

    def _analyse_pipeline_info(self, pipeline_info):
        """
        Calculate pipeline information from raw pipeline data.
        Args:
            pipeline_info (dict): Raw pipeline information.
        Returns:
            dict: Processed pipeline information.
        Raises:
            ProfilerRawFileException: If the format of the input is wrong.
        """
        # Since there may be non-linear pipelines, the processed op info needs to be sorted before final output is
        # produced and saved.
        op_id_info_list = sorted(pipeline_info[0].items(), key=lambda item: item[0])
        return_dict = defaultdict(list)
        dict_opid_parent_id = {}

        for op_id, op_info in op_id_info_list:
            op_name = op_info.get('op_type')[0:-2]
            queue_info_items = self._get_pipeline_metrics_info(op_info.get('metrics', {}))
            children_ids = op_info.get('children', [])
            for child_op_id in children_ids:
                dict_opid_parent_id[child_op_id] = op_id

            return_dict['op_ids'].append(op_id)
            return_dict['op_names'].append(op_name)
            return_dict['pipeline_ops'].append(f'{op_name}(id={op_id})')
            return_dict['num_workers'].append(op_info.get('num_workers'))
            return_dict['queue_average_size'].append(queue_info_items[2])  # output queue average size
            return_dict['queue_utilization_pct'].append(queue_info_items[3])  # output queue utilization percentage
            return_dict['queue_empty_freq_pct'].append(queue_info_items[4])  # output queue empty frequency percentage
            return_dict['children_ids'].append(children_ids)

        return_dict['parent_id'] = [dict_opid_parent_id.get(idx, -1) for idx, _ in op_id_info_list]
        return return_dict

    def _analyse_device_trace_info(self, device_trace_info):
        """
        Calculate device trace information.
        Args:
            device_trace_info (list): Raw device trace information.
        Returns:
            Dictionary consists of:
                per_batch_time: Average per batch time for pipeline in milliseconds
                per_pipeline_time: Average per pipeline time in milliseconds
                per_push_queue_time: Average per queue push time in milliseconds
        """
        q_time = [[], [], []]  # pipeline time, push TDT time, batch time
        prev_time = 0

        for line_data in device_trace_info:
            if len(line_data.split(" ")) < 5:
                logger.warning("Invalid device trace data: %s", line_data)
                continue
            record = [int(d) for d in line_data.split(" ")][0:5]
            if record[2] < 2:  # skip 1st batch
                prev_time = record[4]
                continue
            if record[0] == 0:  # type 0: time record
                q_time[record[1]].append(record[3])
            elif record[0] == 1 and not self._device_queue_file_found:  # type 1: connector size record
                q_time[2].append(record[4] - prev_time)
                prev_time = record[4]

        avg_times = [sum(t) / len(t) if t else -1 for t in q_time]
        return {
            'per_pipeline_time': [round(avg_times[0], 3)],
            'per_push_queue_time': [round(avg_times[1], 3)],
            'per_batch_time': [round(avg_times[2], 3)]
        }

    @staticmethod
    def _get_pipeline_metrics_info(metrics: Dict[str, Any]) -> List[Any]:
        """
        Parse and process the pipeline profiling metrics information for a given op.
        Args:
            metrics (dict): The pipeline profiling metrics information for a given op.
        Returns:
            list: A list containing output queue size, length, average size, utilization percentage,
                  and empty frequency percentage.
        """
        queue_metrics = metrics.get('output_queue', {}) if metrics else {}
        queue_size = queue_metrics.get('size', [])
        queue_length = queue_metrics.get('length', 0)
        if not queue_size:
            return [-1] * 5
        queue_average_size = round(sum(queue_size) / len(queue_size), 2)
        queue_utilization_pct = round(100 * queue_average_size / queue_length, 2) if queue_length else -1
        # Compute percentage of time queue is empty
        queue_empty_freq_pct = round(100 * queue_size.count(0) / len(queue_size), 2)

        return [queue_size, queue_length, queue_average_size, queue_utilization_pct, queue_empty_freq_pct]

    @staticmethod
    def _analyse_cpu_util_info(cpu_util_info: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Calculate CPU utilization information.
        Args:
            cpu_util_info (dict): The CPU utilization profiling information.
        Returns:
            dict: Average CPU utilization percentage for each op, a list ordered by increasing op id
        Raises:
            ProfilerRawFileException: If the format of the CPU utilization file is incorrect.
        """
        cpu_processor_num = cpu_util_info.get('cpu_processor_num')
        cpu_op_info = cpu_util_info.get('op_info')
        if cpu_processor_num is None or not cpu_op_info:
            raise ProfilerRawFileException('The format of MindData CPU utilization JSON file is wrong.')
        for item in cpu_op_info:
            if not item:
                raise ProfilerRawFileException('The contents of MindData CPU utilization JSON file is wrong.')
        # Note: The CPU utilization data may have an extra entry with op_id=-1
        # Omit info for op_id=1
        dict_opid_cpuutil = {}
        for op in cpu_op_info:
            if not op or op["op_id"] == -1:
                continue
            cpu_utilization = []
            for op_sys, op_usr in zip(op["metrics"]["sys_utilization"], op["metrics"]["user_utilization"]):
                cpu_utilization.append(op_sys + op_usr)
            dict_opid_cpuutil[op["op_id"]] = cpu_utilization

        # Initialize oplist_avg_cpu_pct with -1 for each pipeline op, since
        # CPU utilization data may not have information for each pipeline op
        oplist_avg_cpu_pct = [-1] * len(dict_opid_cpuutil)
        for op_id, cpu in dict_opid_cpuutil.items():
            op_avg_cpu_pct = sum(cpu) / len(cpu) if cpu else 0
            oplist_avg_cpu_pct[op_id] = round(op_avg_cpu_pct, 2)

        return {'avg_cpu_pct': oplist_avg_cpu_pct}

    @staticmethod
    def _analyse_composite_info(summary_dict: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Compute composite analysis information from the current summary pipeline data.
        Args:
            summary_dict (dict): Input summary pipeline information.
        Returns:
            dict: Average CPU utilization percentage per worker
        """
        # Build list: average CPU utilization percentage per worker - for each op
        avg_cpu_pct_per_worker = [
            round(c / n if (n != 0 and c >= 0) else -1, 2)
            for c, n in zip(summary_dict.get('avg_cpu_pct', []), summary_dict.get('num_workers', []))
        ]
        return {'avg_cpu_pct_per_worker': avg_cpu_pct_per_worker}

    @staticmethod
    def _analyse_bottleneck_op(summary_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the bottleneck operation using the BottleneckCalculator.
        Args:
            summary_dict (dict): Summary dictionary containing pipeline information.
        Returns:
            Dictionary with the following information, if applicable:
            - CPU utilization analysis
            - queue utilization analysis
            - bottleneck warning: Information on the bottleneck op
                (This is returned only if a potential bottleneck is identified.)
            - bottleneck suggestion: Reason why the subject op is it is identified as
                a potential bottleneck, plus suggestion on how to resolve the bottleneck.
                (This is returned only if a potential bottleneck is identified.)
        """
        try:
            bottleneck_analyzer = BottleneckAnalyzer(summary_dict)
            return bottleneck_analyzer.analyze()
        except IndexError:
            return {}


class BottleneckAnalyzer:
    """ analyzer for bottleneck """

    # These are the threshold values used in the pipeline bottleneck analyzer algorithm
    _THRESHOLDS = {
        '_AVG_CPU_UTIL_PCT_PER_WORKER_MAXIMUM': 75.0,
        '_AVG_CPU_UTIL_PCT_PER_WORKER_MINIMUM': 20.0,
        '_LEAF_OUTPUT_QUEUE_EMPTY_FREQ_PCT_MAXIMUM': 50,
        '_DEVICEQUEUE_INPUT_QUEUE_EMPTY_FREQ_PCT_MAXIMUM': 60,
        '_IN_OUT_QUEUE_UTIL_PCT_DIFF_MAXIMUM': 50,
        '_IN_QUEUE_UTIL_PCT_MAXIMUM': 10
    }

    _NON_MULTITHREADED_OPS = {
        "Barrier", "Concat", "EpochCtrl", "Rename", "Repeat",
        "Shuffle", "Skip", "Take", "Zip"
    }

    def __init__(self, summary_dict):
        self.pipeline_ops = summary_dict["pipeline_ops"]
        self.op_names = summary_dict["op_names"]
        self.op_ids = summary_dict["op_ids"]
        self.num_workers = summary_dict["num_workers"]
        self.queue_average_size = summary_dict["queue_average_size"]
        self.queue_utilization_pct = summary_dict["queue_utilization_pct"]
        self.queue_empty_freq_pct = summary_dict["queue_empty_freq_pct"]
        self.children_ids = summary_dict["children_ids"]
        self.parent_id = summary_dict["parent_id"]
        self.avg_cpu_pct = summary_dict["avg_cpu_pct"]
        self.avg_cpu_pct_per_worker = summary_dict["avg_cpu_pct_per_worker"]
        self.op_id_not_exist = -1
        self.queue_usage_not_exist = -1

    def analyze(self):
        """ analyze all op's usage """
        detailed_analysis = {}
        cpu_analysis = self._analyze_cpu_usage()
        queue_analysis = self._analyze_queue_usage()

        if cpu_analysis:
            detailed_analysis["cpu_analysis_details"] = cpu_analysis
        if queue_analysis:
            detailed_analysis["queue_analysis_details"] = queue_analysis

        bottleneck, suggestion = self._analyze_bottleneck()
        if bottleneck[0]:
            detailed_analysis["bottleneck_warning"] = bottleneck
            detailed_analysis["bottleneck_suggestion"] = suggestion

        return detailed_analysis

    def _analyze_cpu_usage(self):
        """ analyze cpu usage of each op """
        cpu_usage_analysis = []
        for op_id in self.op_ids:
            if op_id == self.op_id_not_exist or self.op_names[op_id] in self._NON_MULTITHREADED_OPS:
                continue

            cpu_pct = self.avg_cpu_pct_per_worker[op_id]
            if cpu_pct > self._THRESHOLDS['_AVG_CPU_UTIL_PCT_PER_WORKER_MAXIMUM'] and self.op_names[op_id]:
                cpu_usage_analysis.append(self._format_high_cpu_usage_suggestion(op_id, cpu_pct))
            elif cpu_pct < self._THRESHOLDS['_AVG_CPU_UTIL_PCT_PER_WORKER_MINIMUM'] and self.num_workers[op_id] > 1:
                cpu_usage_analysis.append(self._format_low_cpu_usage_suggestion(op_id, cpu_pct))
        return cpu_usage_analysis

    def _analyze_queue_usage(self):
        """ analyze queue usage of each op """
        queue_usage_analysis = []
        for op_id in self.op_ids:
            if op_id == self.op_id_not_exist or self.op_names[op_id] in self._NON_MULTITHREADED_OPS:
                continue
            if self.op_names[op_id] == "Batch":
                continue
            in_op_id, out_q = self._get_non_inline_child_recur(op_id), self.queue_utilization_pct[op_id]
            # This is a leaf node since input queue does not exist and output queue exists
            if in_op_id == self.op_id_not_exist and out_q != self.queue_usage_not_exist:
                if out_q <= self._THRESHOLDS['_LEAF_OUTPUT_QUEUE_EMPTY_FREQ_PCT_MAXIMUM']:
                    queue_usage_analysis.append(self._format_leaf_node_suggestion(op_id, out_q))
            # This is device_queue op
            elif self.op_names[op_id] == "DeviceQueue" and in_op_id != self.op_id_not_exist:
                if (self.queue_empty_freq_pct[in_op_id] >
                        self._THRESHOLDS['_DEVICEQUEUE_INPUT_QUEUE_EMPTY_FREQ_PCT_MAXIMUM']):
                    queue_usage_analysis.append(self._format_device_queue_suggestion(op_id, in_op_id))
            elif in_op_id != self.op_id_not_exist and out_q != self.queue_usage_not_exist:
                in_q = self.queue_utilization_pct[in_op_id]
                if (in_q != self.queue_usage_not_exist and
                        in_q - out_q > self._THRESHOLDS['_IN_OUT_QUEUE_UTIL_PCT_DIFF_MAXIMUM']):
                    queue_usage_analysis.append(self._format_internal_node_suggestion(op_id, in_op_id, out_q, in_q))
        return queue_usage_analysis

    def _analyze_bottleneck(self):
        """ analyze bottleneck by using both cpu and queue usage """
        bottleneck, suggestion = "", ""
        for op_id in reversed(self.op_ids):
            if self._should_skip_op(op_id):
                continue

            in_op_id, out_q = self._get_non_inline_child_recur(op_id), self.queue_utilization_pct[op_id]
            wkr_cpu = self.avg_cpu_pct_per_worker[op_id]

            if wkr_cpu > self._THRESHOLDS['_AVG_CPU_UTIL_PCT_PER_WORKER_MAXIMUM']:
                bottleneck = self.pipeline_ops[op_id]
                suggestion = self._format_high_cpu_usage_suggestion(op_id, wkr_cpu)

            elif wkr_cpu < self._THRESHOLDS['_AVG_CPU_UTIL_PCT_PER_WORKER_MINIMUM']:
                in_q_usage = self.queue_utilization_pct[in_op_id]
                if in_op_id != self.op_id_not_exist and (in_q_usage < self._THRESHOLDS['_IN_QUEUE_UTIL_PCT_MAXIMUM']
                                                         or out_q - in_q_usage > self._THRESHOLDS[
                                                             '_IN_OUT_QUEUE_UTIL_PCT_DIFF_MAXIMUM']):
                    bottleneck = self.pipeline_ops[op_id]
                    suggestion = self._format_queue_bottleneck_suggestion(op_id)

        return [bottleneck], [suggestion]

    def _should_skip_op(self, op_id):
        return (op_id == -1 or
                self.op_names[op_id] in self._NON_MULTITHREADED_OPS or
                self.op_names[op_id] == "DeviceQueue")

    def _get_non_inline_child_recur(self, cur_op_id):
        """ get the child id of cur op which isn't an inline op """
        if cur_op_id == self.op_id_not_exist or not self.children_ids[cur_op_id]:
            return self.op_id_not_exist
        cur_child_id = self.children_ids[cur_op_id][0]
        if self.queue_average_size[cur_child_id] != -1:
            return cur_child_id
        return self._get_non_inline_child_recur(cur_child_id)

    def _format_high_cpu_usage_suggestion(self, op_id, cpu_pct):
        return (f"{self.pipeline_ops[op_id]} is using {cpu_pct}% CPU per worker. "
                f"Setting num_parallel_workers>{self.num_workers[op_id]} might bring extra performance.")

    def _format_low_cpu_usage_suggestion(self, op_id, cpu_pct):
        return (f"{self.pipeline_ops[op_id]} is using {cpu_pct}% CPU per worker. "
                f"Using num_parallel_workers={self.num_workers[op_id]} might not bring as much benefit "
                f"due to low CPU usage per worker.")

    def _format_leaf_node_suggestion(self, op_id, out_q):
        return (f"Leaf op {self.pipeline_ops[op_id]} is using {out_q}% of its output queue. "
                f"Setting num_parallel_workers>{self.num_workers[op_id]} might speed up I/O.")

    def _format_device_queue_suggestion(self, op_id, in_op_id):
        return (f"{self.pipeline_ops[op_id]}'s input queue is empty "
                f"{self.queue_empty_freq_pct[in_op_id]}% of the time. "
                f"This might indicate dataset bottlenecks. Hence host cannot keep up with "
                f"the device {self.queue_empty_freq_pct[in_op_id]}% of the time. "
                f"Device waits whenever input queue is empty.")

    def _format_internal_node_suggestion(self, op_id, in_op_id, out_q, in_q):
        return (f"{self.pipeline_ops[op_id]}'s input queue usage={in_q}% is greater than output queue "
                f"usage={out_q}%. This indicates child op {self.pipeline_ops[in_op_id]} "
                f"might be producing faster than its parent {self.pipeline_ops[op_id]} can consume. "
                f"If this op has low CPU utilization, try increasing "
                f"prefetch_size or increasing num_workers.")

    def _format_queue_bottleneck_suggestion(self, op_id):
        return (f"{self.pipeline_ops[op_id]} has low CPU utilization per worker of "
                f"{self.avg_cpu_pct_per_worker[op_id]}% and abnormal queue usage. "
                f"Try increasing prefetch_size.")
