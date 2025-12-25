# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
"""Profiler command executor"""
import shlex
import subprocess
from typing import Union, Tuple

from mindspore import log as logger


class CommandExecutor:
    """
    Execute command in subprocess.
    """
    @staticmethod
    def execute(
            cmd: Union[str, list],
            timeout: int = 10 * 60 * 60,
            check: bool = True,
            encoding: str = "utf-8",
    ) -> Tuple[str, str, int]:
        """
        Execute a shell command with improved error handling and logging.

        Args:
            cmd (Union[str, list]): The command to execute, either as a string or list.
            timeout (int): Maximum execution time in seconds. Defaults to 10 hours.
            check (bool): If True, raise an exception for non-zero exit codes. Defaults to True.
            encoding (str): Specify the encoding for output. Defaults to 'utf-8'.

        Returns:
            Tuple[str, str, int]: A tuple containing (stdout, stderr, return_code).

        Raises:
            RuntimeError: If any error occurs during command execution.
        """
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=check,
                shell=False,
                encoding=encoding,
            )
            logger.info(
                f"Command '{cmd}' executed successfully. Return code: {result.returncode}"
            )
            return result.stdout, result.stderr, result.returncode

        except subprocess.TimeoutExpired as e:
            error_msg = f"Command '{cmd}' timed out after {timeout} seconds"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except subprocess.CalledProcessError as e:
            error_msg = (
                f"Command '{cmd}' failed with return code {e.returncode}, "
                f"Standard Error: {e.stderr}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except (FileNotFoundError, PermissionError) as e:
            error_msg = f"Error executing command '{cmd}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e: # pylint: disable=broad-exception-caught
            error_msg = (
                f"Unexpected error occurred while executing command '{cmd}': {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
