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
"""Ascend communication viewer"""
import os
import re
from collections import defaultdict

from typing import Dict
from mindspore.profiler.analysis.viewer.base_viewer import BaseViewer
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.log import ProfilerLogger

from mindspore import log as logger
from mindspore.profiler.common.constant import JitLevel


class AscendCommunicationViewer(BaseViewer):
    """Ascend communication viewer"""

    COMMUNICATION_TIME_INFO = "Communication Time Info"
    START_TIMESTAMP = "Start Timestamp(us)"
    COMMUNICATION_BANDWIDTH_INFO = "Communication Bandwidth Info"
    HCOM_SEND = "Send"
    HCOM_RECEIVE = "Receive"
    TOTAL = "Total"
    SYNCHRONIZATION_TIME_RATIO = "Synchronization Time Ratio"
    SYNCHRONIZATION_TIME_MS = "Synchronization Time(ms)"
    WAIT_TIME_RATIO = "Wait Time Ratio"
    TRANSIT_TIME_MS = "Transit Time(ms)"
    TRANSIT_SIZE_MB = "Transit Size(MB)"
    SIZE_DISTRIBUTION = "Size Distribution"
    WAIT_TIME_MS = "Wait Time(ms)"
    BANDWIDTH_GB_S = "Bandwidth(GB/s)"
    COMMUNICATION = "communication.json"
    COMMUNICATION_MATRIX = "communication_matrix.json"
    P2P = "p2p"
    COLLECTIVE = "collective"
    TRANSPORT_TYPE = "Transport Type"
    PATTERN1 = re.compile(r"receive|send")
    PATTERN2 = re.compile(r"invalid|broadcast|allreduce|reduce|"
                          r"allgather|reducescatter|scatter|alltoall|alltoallv|alltoallvc")

    def __init__(self, **kwargs):
        super().__init__()
        self.step_list = []
        self.output_communication = {}
        self.output_matrix_data = {}
        self._output_path = kwargs.get("ascend_profiler_output_path")
        self._msprof_analyze_output_path = kwargs.get("msprof_analyze_output_path")
        self._ascend_ms_dir = kwargs.get("ascend_ms_dir")
        self._is_set_schedule = kwargs.get("is_set_schedule")
        self._jit_level = kwargs.get("jit_level")
        ProfilerLogger.init(self._ascend_ms_dir)
        self._logger = ProfilerLogger.get_instance()
        self._communication_input_path = os.path.join(
            self._msprof_analyze_output_path,
            self.COMMUNICATION
        )
        self._communication_matrix_input_path = os.path.join(
            self._msprof_analyze_output_path,
            self.COMMUNICATION_MATRIX
        )
        self._communication_output_path = os.path.join(
            self._output_path,
            self.COMMUNICATION
        )
        self._communication_matrix_output_path = os.path.join(
            self._output_path,
            self.COMMUNICATION_MATRIX
        )

    def save(self, data=None):
        """
        Save ascend integrate data.
        """
        self._logger.info("AscendCommunicationViewer start")
        try:
            self._init_step_list(data)
            self._generate_communication()
            self._generate_matrix()
            self._save_analyze_data()
        except Exception as e:  # pylint: disable=W0703
            self._logger.error("Failed to save ascend communication data, error: %s", e, exc_info=True)
        self._logger.info("AscendCommunicationViewer end")

    def _init_step_list(self, data):
        """
        Init step list.
        """
        trace_container = data.get("trace_view_container", None)
        if trace_container is None:
            raise ValueError("trace view container is None")
        step_id_to_time_dict = trace_container.get_step_id_time_dict()

        if not self._is_set_schedule or self._jit_level == JitLevel.GRAPH_LEVEL or not step_id_to_time_dict:
            self._update_default_step_list()
        else:
            self._update_step_list(step_id_to_time_dict)

    def _update_default_step_list(self):
        """
        When the step dict is empty, it is set to the default value.
        """
        self.step_list = [{"step_id": "0", "start_ts": 0, "end_ts": float('inf'), "comm_ops": {}}]

    def _update_step_list(self, step_id_to_time_dict: Dict):
        """
        When the step dict is not empty, set a value that contains the step id.
        """
        for step_id, (start_time, end_time) in step_id_to_time_dict.items():
            step_dict = {
                "step_id": step_id,
                "start_ts": start_time,
                "end_ts": end_time,
                "comm_ops": {}
            }
            self.step_list.append(step_dict)

    def _save_analyze_data(self):
        """
        Save analyse data
        """
        self._logger.info("Save ascend communication data start")
        if not self.output_communication:
            return
        FileManager.create_json_file(self._communication_output_path, self.output_communication)
        self._logger.info("Save ascend communication data done")
        if not self.output_matrix_data:
            return
        FileManager.create_json_file(self._communication_matrix_output_path, self.output_matrix_data)
        self._logger.info("Save ascend communication matrix data done")

    @staticmethod
    def _combine_size_distribution(op_dict: dict, total_dict: dict):
        """combine size distribution"""
        for size, size_info in op_dict.items():
            total_dict[size][0] += size_info[0]
            total_dict[size][1] += size_info[1]

    @staticmethod
    def _compute_ratio(dividend: float, divisor: float):
        """compute ratio"""
        if abs(divisor) < 1e-15:
            return 0
        return round(dividend / divisor, 4)

    def _generate_communication(self):
        """
        generate communication.json
        """
        if not os.path.exists(self._communication_input_path):
            return
        communication_data = FileManager.read_json_file(self._communication_input_path)
        if not communication_data:
            return
        self._split_comm_op_by_step(communication_data)

        for step_info in self.step_list:
            step = "step" + step_info.get("step_id") if step_info.get("step_id") else "step"
            self.output_communication[step] = self._get_communication_ops_dict(step_info.get("comm_ops"))

    def _generate_matrix(self):
        """generate matrix"""
        if not os.path.exists(self._communication_matrix_input_path):
            return
        matrix_data = FileManager.read_json_file(self._communication_matrix_input_path)
        if not matrix_data:
            return
        matrix_data_by_step = self._split_matrix_by_step(matrix_data)

        for step, comm_matrix_data in matrix_data_by_step.items():
            self.output_matrix_data[step] = self._get_matrix_ops_dict(comm_matrix_data)

    def _split_comm_op_by_step(self, communication_data: dict):
        """split comm op by step"""
        if len(self.step_list) == 1:
            self.step_list[0]["comm_ops"] = communication_data
        for communication_op, communication_op_info in communication_data.items():
            start_time = communication_op_info.get(self.COMMUNICATION_TIME_INFO, {}).get(self.START_TIMESTAMP)
            for step_info in self.step_list:
                if step_info.get("start_ts", -1) <= start_time <= step_info.get("end_ts", -1):
                    step_info.get("comm_ops", {})[communication_op] = communication_op_info
                    break

    def _split_communication_p2p_ops(self, op_data: dict):
        """
        split communicate
        """
        comm_op_dict = {self.P2P: {}, self.COLLECTIVE: {}}
        for communication_op, communication_info in op_data.items():
            if communication_op.find(self.HCOM_SEND) != -1 or communication_op.find(self.HCOM_RECEIVE) != -1:
                comm_op_dict[self.P2P][communication_op] = communication_info
            elif communication_op.startswith(self.TOTAL):
                continue
            else:
                comm_op_dict[self.COLLECTIVE][communication_op] = communication_info
        return comm_op_dict

    def _split_matrix_by_step(self, matrix_data: dict) -> dict:
        """
        split matrix by step
        """
        matrix_data_by_step = {}
        if self._is_step_list_empty():
            matrix_data_by_step["step"] = matrix_data
            return matrix_data_by_step

        for comm_op in matrix_data:
            for step_info in self.step_list:
                if comm_op in step_info.get("comm_ops", {}):
                    step = "step" + step_info.get("step_id") if step_info.get("step_id") else "step"
                    matrix_data_by_step.setdefault(step, {})[comm_op] = matrix_data.get(comm_op)
                    break
        return matrix_data_by_step

    def _get_communication_ops_dict(self, op_data: dict) -> dict:
        """get communication ops dict"""
        comm_op_dict = self._split_communication_p2p_ops(op_data)
        self._compute_total_info(comm_op_dict[self.P2P])
        self._compute_total_info(comm_op_dict[self.COLLECTIVE])
        return comm_op_dict

    def _integrate_matrix_data(self, comm_op_dict_simple):
        """integrate the matrix data"""
        comm_op_dict = defaultdict(dict)
        for new_comm_op_name, data in comm_op_dict_simple.items():
            data.sort(key=lambda x: x[self.BANDWIDTH_GB_S], reverse=True)
            t_type = data[0].get(self.TRANSPORT_TYPE, '')
            t_size = sum(x.get(self.TRANSIT_SIZE_MB, 0) for x in data)
            t_time = sum(x.get(self.TRANSIT_TIME_MS, 0) for x in data)
            bandwidth = self._compute_ratio(t_size, t_time)

            link = new_comm_op_name[2]

            comm_op_dict[f'{new_comm_op_name[0]}-top1@{new_comm_op_name[1]}'].update({link: data[0]})
            comm_op_dict[f'{new_comm_op_name[0]}-middle@{new_comm_op_name[1]}'].update({link: data[len(data) // 2]})
            comm_op_dict[f'{new_comm_op_name[0]}-bottom1@{new_comm_op_name[1]}'].update({link: data[-1]})
            index2 = -2
            index3 = -3
            if len(data) == 1:
                index2 = -1
                index3 = -1
            elif len(data) == 2:
                index3 = -2
            comm_op_dict[f'{new_comm_op_name[0]}-bottom2@{new_comm_op_name[1]}'].update({link: data[index2]})
            comm_op_dict[f'{new_comm_op_name[0]}-bottom3@{new_comm_op_name[1]}'].update({link: data[index3]})
            comm_op_dict[f'{new_comm_op_name[0]}-total@{new_comm_op_name[1]}'].update({link: {
                self.TRANSPORT_TYPE: t_type,
                self.TRANSIT_SIZE_MB: t_size,
                self.TRANSIT_TIME_MS: t_time,
                self.BANDWIDTH_GB_S: bandwidth
            }})
        return comm_op_dict

    def _get_matrix_ops_dict(self, op_data: dict) -> dict:
        """parse matrix data"""
        comm_op_dict_simple_p2p = defaultdict(list)
        comm_op_dict_simple_collective = defaultdict(list)

        for communication_op, communication_info in op_data.items():
            if communication_op.find(self.HCOM_SEND) != -1 or communication_op.find(self.HCOM_RECEIVE) != -1:

                match_obj = self.PATTERN1.search(communication_op.lower())
                comm_op_type = match_obj.group()
                for link, data in communication_info.items():
                    new_comm_op_name = (comm_op_type, communication_op.split("@")[-1], link)
                    data['op_name'] = communication_op.split("@")[0]
                    comm_op_dict_simple_p2p[new_comm_op_name].append(data)

            elif communication_op.startswith(self.TOTAL):
                continue
            else:
                match_obj = self.PATTERN2.search(communication_op.lower())
                if not match_obj:
                    comm_op_type = communication_op.lower().split('/')[-1].split('-op')[0]
                    logger.warning(
                        "Communication operator type not found communication_op: %s, use comm_op_type: %s",
                        communication_op, comm_op_type)
                else:
                    comm_op_type = match_obj.group()
                for link, data in communication_info.items():
                    new_comm_op_name = (comm_op_type, communication_op.split("@")[-1], link)
                    data['op_name'] = communication_op.split("@")[0]
                    comm_op_dict_simple_collective[new_comm_op_name].append(data)

        comm_op_dict = {self.P2P: self._integrate_matrix_data(comm_op_dict_simple_p2p),
                        self.COLLECTIVE: self._integrate_matrix_data(comm_op_dict_simple_collective)}

        return comm_op_dict

    def _is_step_list_empty(self):
        """is step list empty"""
        for step_info in self.step_list:
            if step_info.get("comm_ops"):
                return False
        return True

    def _compute_total_info(self, comm_ops: dict):
        """
        compute total info
        """
        if not comm_ops:
            return
        total_time_info_dict = defaultdict(float)
        total_bandwidth_info_dict = {}
        for _, communication_op_info in comm_ops.items():
            for com_info, com_info_dict in communication_op_info.items():
                if com_info == self.COMMUNICATION_TIME_INFO:
                    self._combine_time_info(com_info_dict, total_time_info_dict)
                if com_info == self.COMMUNICATION_BANDWIDTH_INFO:
                    self._combine_bandwidth_info(com_info_dict, total_bandwidth_info_dict)
        self._compute_time_ratio(total_time_info_dict)
        self._compute_bandwidth_ratio(total_bandwidth_info_dict)
        comm_ops['Total Op Info'] = {
            self.COMMUNICATION_TIME_INFO: total_time_info_dict,
            self.COMMUNICATION_BANDWIDTH_INFO: total_bandwidth_info_dict
        }

    def _combine_time_info(self, com_info_dict: dict, total_time_info_dict: dict):
        """combine time info"""
        ratio_list = [self.WAIT_TIME_RATIO, self.SYNCHRONIZATION_TIME_RATIO]
        for time_info in com_info_dict:
            if time_info not in ratio_list and time_info != self.START_TIMESTAMP:
                total_time_info_dict[time_info] += com_info_dict.get(time_info)

    def _combine_bandwidth_info(self, com_info_dict: dict, total_bandwidth_info_dict: dict):
        """
        combine bandwidth info
        """
        add_list = [self.TRANSIT_TIME_MS, self.TRANSIT_SIZE_MB]
        dict_list = [self.SIZE_DISTRIBUTION]
        for transport_type, part_transport_dict in com_info_dict.items():
            if transport_type not in total_bandwidth_info_dict:
                total_bandwidth_info_dict[transport_type] = {
                    self.TRANSIT_TIME_MS: 0,
                    self.TRANSIT_SIZE_MB: 0,
                    self.SIZE_DISTRIBUTION: defaultdict(lambda: [0, 0])
                }
            for bandwidth_msg, value in part_transport_dict.items():
                if bandwidth_msg in add_list:
                    total_bandwidth_info_dict[transport_type][bandwidth_msg] += value
                if bandwidth_msg in dict_list:
                    self._combine_size_distribution(
                        value,
                        total_bandwidth_info_dict.get(transport_type, {})
                        .get(bandwidth_msg, defaultdict(lambda: [0, 0]))
                    )

    def _compute_time_ratio(self, total_time_info_dict: dict):
        """compute time ratio"""
        total_time_info_dict[self.WAIT_TIME_RATIO] = \
            self._compute_ratio(total_time_info_dict.get(self.WAIT_TIME_MS, 0),
                                total_time_info_dict.get(self.WAIT_TIME_MS, 0) +
                                total_time_info_dict.get(self.TRANSIT_TIME_MS, 0))
        total_time_info_dict[self.SYNCHRONIZATION_TIME_RATIO] = \
            self._compute_ratio(total_time_info_dict.get(self.SYNCHRONIZATION_TIME_MS, 0),
                                total_time_info_dict.get(self.TRANSIT_TIME_MS, 0) +
                                total_time_info_dict.get(self.SYNCHRONIZATION_TIME_MS, 0))

    def _compute_bandwidth_ratio(self, total_bandwidth_info_dict: dict):
        """compute bandwidth ratio"""
        for _, bandwidth_dict in total_bandwidth_info_dict.items():
            self._compute_ratio(bandwidth_dict.get(self.TRANSIT_SIZE_MB, 0),
                                bandwidth_dict.get(self.TRANSIT_TIME_MS, 0))
