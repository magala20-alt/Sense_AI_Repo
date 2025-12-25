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
"""mindspore utils runtime order check."""
import csv
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import List, Dict, Union, Optional
import sys
import mindspore.log as logger
from mindspore._c_expression import CommExecOrderChecker

# Set Recursion Depth Limit
sys.setrecursionlimit(10000)
# support hccl group 150000 card
csv.field_size_limit(1024 * 1024 * 10)


def comm_exec_order_check(action):
    """
    Call the CommExecOrderCheck class to start the collection of communication operator execution sequences
    or stop the collection and validate the execution order.

    Args:
        action (str): Control command - 'start' to begin collection, 'end' to stop and validate.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.utils import comm_exec_order_check
        >>> comm_exec_order_check("start")
        >>> model.train(1, train_dataset)
        >>> comm_exec_order_check("end")
    """
    if not isinstance(action, str):
        raise TypeError("The 'action' parameter must be a string.")
    checker = CommExecOrderCheck()
    checker(action)


class CommExecOrderCheck:
    """Controller for communication execution order verification.

    Provides interface for starting/stopping the collection of communication
    operator execution sequences. Integrates with C++ backend for actual
    order tracking.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.action = None
            self.order_checker = CommExecOrderChecker.get_instance()
            self.is_collecting = False
            self.initialized = True

    def __call__(self, action):
        """
        Args:
            action (str): Control command - 'start' to begin collection,
                        'end' to stop and validate
        """
        self.action = action
        if action == "start":
            self.start_function()
        elif action == "end":
            self.end_function()
        else:
            raise ValueError("Invalid action. Please use 'start' or 'end'.")

    def start_function(self):
        if self.is_collecting:
            logger.error("The 'start' action cannot be called twice.")
            return
        self.is_collecting = True
        self.order_checker.start_collect_exec_order()

    def end_function(self):
        if not self.is_collecting:
            logger.error("The 'end' action cannot be called before the 'start' action.")
            return
        self.is_collecting = False
        self.order_checker.stop_collect_exec_order()


class ExecuteOrder:
    """Represents a single record from the execute_order.csv file."""

    def __init__(self, index: str, group: str, comm_rank: str,
                 primitive: str, src_rank: str = '', dest_rank: str = '', root_rank: str = '', input_shape: str = '',
                 output_shape: str = ''):
        self.index = index
        self.group = group
        self.comm_rank = comm_rank.split()  # Split comm_rank into a list of individual ranks
        self.primitive = primitive
        self.src_rank = src_rank
        self.dest_rank = dest_rank
        self.root_rank = root_rank
        self.input_shape = input_shape
        self.output_shape = output_shape

    def generate_base_key(self, rank: str) -> str:
        """
        Generates a unique base key for counting, excluding the counter part.
        """
        if not self.src_rank and not self.dest_rank and not self.root_rank:
            # Aggregate communication status, for example, allGather.
            comm_str = ",".join(self.comm_rank)
            return f"{self.primitive}_{self.group}_({comm_str})"

        if self.primitive in ["Send", "DistCommIsend", "InnerCommIsend"]:
            # Unique base key of the Send operation.
            return f"Send_Receive_{self.group}_({rank})->({self.dest_rank})_{self.input_shape}"

        if self.primitive in ["Receive", "DistCommIrecv", "InnerCommIrecv"]:
            # Unique base key of the Recv operation
            return f"Send_Receive_{self.group}_({self.src_rank})->({rank})_{self.output_shape}"

        # Other operations, such as broadCast
        parts = [f"{self.primitive}_{self.group}", f"commRank:({','.join(self.comm_rank)})"]
        if self.src_rank:
            parts.append(f"srcRank:({self.src_rank})")
        if self.dest_rank:
            parts.append(f"destRank:({self.dest_rank})")
        if self.root_rank:
            parts.append(f"rootRank:({self.root_rank})")

        return "_".join(parts)

    def generate_key(self, rank: str, count: int) -> str:
        """
        Generates the final unique key, including the counterpart.
        """
        base_key = self.generate_base_key(rank)
        return f"{base_key}_{count}th"


class RankFolderParser:
    """
    Parses specified folder(s) and reads execute_order.csv files within rank_x folders.
    """

    REQUIRED_COLUMNS = [
        "index", "group", "comm_rank", "primitive",
        "src_rank", "dest_rank", "root_rank", "input_shape", "input_type", "output_shape", "output_type", "input_size",
        "output_size"
    ]

    def __init__(self, folders: Union[str, List[str]]):
        # Ensure folders is always a list of absolute paths
        if isinstance(folders, str):
            self.folders = [folders]
        else:
            self.folders = folders

        # Validate provided folders or rank_{x} paths
        self._validate_paths()

        # Store results here
        self.result_map: Dict[str, List[ExecuteOrder]] = {}

        # Determine optimal number of workers based on CPU count
        self.cpu_cores = cpu_count()
        self.max_threads = min(8, self.cpu_cores * 2)  # For I/O-bound tasks
        self.max_processes = min(8, self.cpu_cores)  # For CPU-bound tasks

    def _validate_paths(self):
        """
        Validate all provided paths.
        Each path must be either:
        - A valid absolute directory containing rank_{x} folders, or
        - A valid absolute path directly pointing to a rank_{x} folder.

        Additionally, checks for duplicate rank_{x} folders and raises an error if duplicates are found.
        """
        seen_ranks = set()  # To track rank_x folder names and detect duplicates

        for path in self.folders:
            if not os.path.isabs(path):
                raise ValueError(
                    f"Invalid path: {path}. "
                    f"Please provide an absolute path, e.g., '/absolute/path/to/folder' or '/absolute/path/to/rank_x'."
                )
            if not os.path.exists(path):
                raise ValueError(
                    f"Path does not exist: {path}. "
                    f"Ensure the path exists and is accessible."
                )

            # Check if it is a specific rank_{x} folder
            if os.path.isdir(path) and os.path.basename(path).startswith("rank_"):
                rank_name = os.path.basename(path)
                if rank_name in seen_ranks:
                    raise ValueError(
                        f"Duplicate rank folder detected: {rank_name}. "
                        f"Each rank_x folder must be unique. Please remove duplicates."
                    )
                seen_ranks.add(rank_name)
                continue  # Valid rank_{x} folder

            # Check if it is a directory containing rank_{x} folders
            if os.path.isdir(path):
                rank_folders = [d for d in os.listdir(path) if
                                d.startswith("rank_") and os.path.isdir(os.path.join(path, d))]
                if not rank_folders:
                    raise ValueError(
                        f"No rank_x folders found in {path}. Ensure the directory contains at least one folder named "
                        f"'rank_x', where x is an integer."
                    )

                # Check for duplicates within this directory
                for rank_folder in rank_folders:
                    if rank_folder in seen_ranks:
                        raise ValueError(
                            f"Duplicate rank folder detected: {rank_folder} in {path}. "
                            f"Each rank_x folder must be unique. Please remove duplicates."
                        )
                    seen_ranks.add(rank_folder)
                continue  # Valid directory containing rank_{x} folders

            # Invalid case
            raise ValueError(
                f"Invalid path: {path}. "
                f"Paths must either be rank_x folders or directories containing rank_x folders."
            )

    def parse(self) -> Dict[str, List[ExecuteOrder]]:
        """
        Main parsing function using a ProcessPoolExecutor to handle multiple paths.
        Each rank_{x} folder is processed in parallel.
        """
        with ProcessPoolExecutor(max_workers=self.max_processes) as process_executor:
            futures = [process_executor.submit(self._parse_path, path) for path in self.folders]
            for future in as_completed(futures):
                try:
                    path_result = future.result()
                    if path_result:
                        self.result_map.update(path_result)
                except FileNotFoundError as e:
                    logger.error(f"File not found: {e}. Please ensure all required files are present.")
                except ValueError as e:
                    logger.error(f"Value error: {e}. Please check the file contents or paths.")

        return self.result_map

    def _parse_path(self, path: str) -> Dict[str, List]:
        """
        Helper function to parse a single path. Handles both:
        - Direct rank_{x} folder paths.
        - Parent directories containing multiple rank_{x} folders.
        """
        result = {}

        # If the path is a rank_{x} folder, parse it directly
        if os.path.basename(path).startswith("rank_"):
            rank_id = os.path.basename(path).split("_")[1]
            # Adding one more layer to access the "execute_order" folder
            execute_order_path = os.path.join(path, "execute_order")
            if not os.path.exists(execute_order_path):
                raise FileNotFoundError(
                    f"Execute order folder does not exist: {execute_order_path} "
                    f"for rank_{rank_id} folder."
                )
            rank_result = self.parse_rank_folder(execute_order_path, rank_id)
            if rank_result:
                result[rank_id] = rank_result[1]  # Extract execute orders
            return result

        # If the path is a directory containing rank_{x} folders, parse all
        with ThreadPoolExecutor(max_workers=self.max_threads) as thread_executor:
            futures = []
            for d in os.listdir(path):
                if d.startswith("rank_"):
                    rank_id = d.split("_")[1]
                    rank_folder_path = os.path.join(path, d)
                    execute_order_path = os.path.join(rank_folder_path, "execute_order")

                    if not os.path.exists(execute_order_path):
                        raise FileNotFoundError(
                            f"Execute order folder does not exist: {execute_order_path} "
                            f"for rank_{rank_id} folder."
                        )
                    futures.append(thread_executor.submit(self.parse_rank_folder, execute_order_path, rank_id))

            for future in as_completed(futures):
                try:
                    rank_id, execute_orders = future.result()
                    if execute_orders:
                        result[rank_id] = execute_orders
                except FileNotFoundError as e:
                    logger.error(f"File not found during parallel processing: {e}. "
                                 f"Ensure the required files are present.")
                except ValueError as e:
                    logger.error(f"Value error during parallel processing: {e}. Check file format or contents.")

        return result

    def parse_rank_folder(self, rank_folder_path: str, rank_id: str) -> Optional[tuple]:
        """
        Parse a single rank_{x} folders execute_order.csv file with header validation.
        """
        execute_order_file = os.path.join(rank_folder_path, "comm_execute_order.csv")

        if not os.path.exists(execute_order_file):
            logger.error(
                f"No execute_order.csv found in {rank_folder_path}. Skipping this folder. "
                f"Ensure the rank_{rank_id} folder contains a valid comm_execute_order.csv file."
            )
            return rank_id, None

        execute_orders = []

        try:
            with open(execute_order_file, mode='r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)

                # Validate the header
                if csv_reader.fieldnames != self.REQUIRED_COLUMNS:
                    logger.error(
                        f"Invalid header in {execute_order_file}. Skipping this file. "
                        f"Expected columns: {self.REQUIRED_COLUMNS}. "
                        f"Ensure the file contains the correct column names."
                    )
                    return rank_id, None

                # Read and parse data rows
                for row in csv_reader:
                    execute_order = ExecuteOrder(
                        index=row["index"],
                        group=row["group"],
                        comm_rank=row["comm_rank"],
                        primitive=row["primitive"],
                        src_rank=row["src_rank"],
                        dest_rank=row["dest_rank"],
                        root_rank=row["root_rank"],
                        input_shape=row["input_shape"],
                        output_shape=row["output_shape"],

                    )
                    execute_orders.append(execute_order)
        except FileNotFoundError as e:
            logger.error(f"File not found: {execute_order_file}. Ensure the file exists and is accessible. Error: {e}")
            return rank_id, None

        return rank_id, execute_orders


def modify_execute_orders(execute_orders_map: dict) -> dict:
    """
    Modify and generate unique execution order keys for each rank.

    This function processes a mapping of execution orders grouped by ranks. For each order,
    it generates a unique key by combining a base key and a counter, ensuring all orders
    are uniquely identifiable. The result is a dictionary where the keys are rank identifiers
    and the values are lists of unique execution order keys.

    Args:
        execute_orders_map (dict): A dictionary where keys are rank identifiers (e.g., "rank_0")
                                and values are lists of ExecuteOrder objects. If a rank has no
                                orders, its value may be `None`.

    Returns:
        dict: A dictionary where keys are rank identifiers and values are lists of unique string
            keys representing the modified execution orders for each rank.
    """
    result = {}

    for rank, execute_orders in execute_orders_map.items():
        # If execute_orders is None, use an empty list instead.
        if execute_orders is None:
            execute_orders = []

        count_map = defaultdict(int)
        new_orders = []

        for order in execute_orders:
            # Use generate_base_key to generate the base key for counting.
            base_key = order.generate_base_key(rank)

            count_map[base_key] += 1
            count = count_map[base_key]

            # Use generate_key to generate the final unique key.
            new_key = order.generate_key(rank, count)
            new_orders.append(new_key)

        # Save the generated new order list to the result dictionary.
        result[rank] = new_orders

    return result


def parse_and_validate(data: dict, all_rank: bool = True):
    """
    Parse and validate execution orders in a directed graph structure.

    This function checks the integrity and consistency of a given dataset, ensuring all required
    keys are present and correctly referenced. It also validates the structure of the input data
    and parses string values to extract meaningful components.

    Args:
        data (dict): A dictionary where keys are string identifiers and values are lists of strings.
                    Each value represents a dependency or reference to other keys.
        all_rank (bool): If True, checks that all elements referenced in the data are present as keys
                        in the dictionary. If False, only checks intersections.

    Returns:
        None: Log error messages to the console if validation fails, otherwise completes silently.

    Raises:
        ValueError: Raised indirectly if `parse_elements` encounters malformed input strings.
        TypeError: Raised indirectly if data contains unexpected types.
    """
    def parse_elements(value: str, max_groups: int = 2) -> set:
        """Extract unique elements inside the first one or two parentheses from a string."""
        groups = re.findall(r'\((.*?)\)', value)
        limited_groups = groups[:max_groups]  # Limit to the first `max_groups` matches
        return {item.strip() for group in limited_groups for item in group.split(',')}

    if not isinstance(data, dict):
        logger.error("Input must be a dictionary with string keys and lists of strings as values.")
        return

    key_to_values = {key: set(values) for key, values in data.items() if
                     isinstance(values, list) and all(isinstance(v, str) for v in values)}

    for key, values in data.items():
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            logger.error(f"Values for key '{key}' must be a list of strings.")
            continue

        for value in values:
            try:
                elements = parse_elements(value)
            except (ValueError, TypeError, AttributeError) as e:
                logger.error(f"Unable to parse elements from value '{value}' in key '{key}'. Error: {e}")
                continue

            # Check for missing keys if all_rank is True
            if all_rank:
                missing_keys = elements - key_to_values.keys()
                if missing_keys:
                    logger.error(f"The following keys are missing for value '{value}': {missing_keys}")
                    continue

            # Check if the value is present in the referenced keys
            for element in elements & key_to_values.keys() if not all_rank else elements:
                if value not in key_to_values[element]:
                    logger.error(f"Key '{element}' is missing the value '{value}'.")


def detect_cycle_in_graph(ranks_map):
    """
    Detects a cycle in the directed graph constructed from ranks_map.

    Args:
    - ranks_map (dict): A dictionary where keys are rank names and values are lists of nodes.

    Returns:
    - tuple: (cycle_path, cycle_ranks) where cycle_path is a list of nodes forming the cycle and cycle_ranks
                is a list of rank transitions corresponding to the cycle path.
    """
    graph = defaultdict(list)
    rank_edges = {}

    for rank, nodes in ranks_map.items():
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            graph[u].append(v)
            rank_edges[(u, v)] = rank

    visited = set()
    path = []
    node_indices = {}
    cycle_path = []
    cycle_ranks = []

    # Use a stack to simulate recursion
    stack = []
    for node in list(graph.keys()):
        if node not in visited:
            stack.append((node, False))  # (node, is_processed)

            while stack:
                current_node, is_processed = stack.pop()

                if is_processed:
                    # Post-processing: remove node from path and indices
                    path.pop()
                    del node_indices[current_node]
                    continue

                if current_node in node_indices:
                    # Found a cycle
                    cycle_start = node_indices[current_node]
                    cycle_path = path[cycle_start:] + [current_node]
                    for i in range(cycle_start, len(path)):
                        u = path[i]
                        v = path[i + 1] if i + 1 < len(path) else current_node
                        cycle_ranks.append(f"{rank_edges[(u, v)]} {u} -> {v}")
                    return cycle_path, cycle_ranks

                if current_node in visited:
                    continue

                visited.add(current_node)
                node_indices[current_node] = len(path)
                path.append(current_node)

                # Mark current node as processed
                stack.append((current_node, True))

                # Add neighbors to stack
                for neighbor in reversed(graph[current_node]):
                    stack.append((neighbor, False))

    return None, None


def determine_all_rank(folders_):
    """
    Determine the value of all_rank based on the input folders_.

    Args:
        folders_ (str | list): Folder path(s) to process.

    Returns:
        bool | None: Returns True/False for valid inputs, or None for invalid inputs.
    """
    if isinstance(folders_, str) and folders_:
        return True
    if isinstance(folders_, list):
        if len(folders_) == 1:
            return True
        if len(folders_) > 1:
            return False
        return None
    return None


def output_cycle_results(cycle_path, cycle_ranks):
    """
    Helper function to output cycle detection results.

    Args:
        cycle_path (list): List of nodes forming a cycle, if any.
        cycle_ranks (list): List of ranks involved in the cycle.

    Returns:
        None: Outputs results to the console.
    """
    if cycle_path:
        logger.error("Cycle detected:")
        logger.error(" -> ".join(cycle_path) + f" -> {cycle_path[0]}")  # Close the cycle
        logger.error("Involving ranks:")
        for rank in cycle_ranks:
            logger.error(rank)
    else:
        logger.warning("Cycle Check success. There is no cycle in the graph.")


def runtime_execution_order_check(folders_, all_rank=None):
    """
    Verify the rank_x folder in the specified directory.

    Parameter description:
    1. folders (str or list[str]):
    - Can be a single string or a list of strings representing the rank_x folder path or its upper-level directory path.
    - If the input directory is the upper-level directory path, the function automatically identifies and verifies
    all rank_x folders in the directory.
    - If a specific rank_x folder path is passed, the function will only verify these specified folders.

    2. check_all (optional, true by default):
    - Controls whether to verify all rank_x folders.
    - True (default): Verify all files.
    - False: Only part of the transferred rank_x folder is verified.

    Example:
    Example 1: Verify all rank_x folders (default behavior)
    runtime_execution_order_check("path/to/parent_folder")

    Example 2: Verify only some specified rank_x folders.
    runtime_execution_order_check(["path/to/rank_1", "path/to/rank_2"], all_rank=False)

    Example 3: Verify a single rank_x folder.
    runtime_execution_order_check("path/to/rank_x", all_rank=True)

    Precautions:
    - When folders is the upper-level directory path, the function automatically
    searches for and verifies the rank_x folder.
    - When check_all is set to false, only the rank_x folder specified in the input path is verified.
    """
    # Use the provided all_rank if available, otherwise determine it
    if all_rank is None:
        all_rank = determine_all_rank(folders_)

    if folders_ is None:  # Input validation failed
        logger.error("Invalid input. `folders_` must be a non-empty string or a list with at least one string element.")
        return

    # Parse folders
    parser = RankFolderParser(folders_)
    result_map = parser.parse()

    if not result_map:
        logger.error("No valid rank data found. Execution order check aborted.")
        return

    # Check for any rank with empty execution orders
    for rank, orders in result_map.items():
        if not orders:
            logger.error(f"Rank {rank} has no valid execution orders. Please check the csv file.")
            return

    # Modify execution orders
    modified_orders = modify_execute_orders(result_map)

    # Parse and validate execution orders
    parse_and_validate(modified_orders, all_rank)

    # Detect cycles
    cycle_path, cycle_ranks = detect_cycle_in_graph(modified_orders)

    # Output results
    output_cycle_results(cycle_path, cycle_ranks)
