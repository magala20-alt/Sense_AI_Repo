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
"""Base parser"""
from typing import (
    Any,
    List,
    Callable,
    Optional,
)
from multiprocessing import Process
from mindspore import log as logger
from mindspore.profiler.common.log import ProfilerLogger


class BaseParser:
    """
    Base class for all parsers in the workflow.
    """
    EXEC_HOOK_TIMEOUT = 30 * 60

    def __init__(self, next_parser: Optional["BaseParser"] = None):
        """
        Initialize the BaseParser.

        Args:
            next_parser (Optional[BaseParser]): The next parser in the chain.
        """
        self.next_parser: Optional["BaseParser"] = next_parser
        self._post_hooks: List[Callable[[Any], None]] = []
        self._logger = ProfilerLogger.get_instance()

    def set_next(self, next_parser: "BaseParser") -> "BaseParser":
        """
        Set the next parser in the chain.

        Args:
            next_parser (BaseParser): The next parser to be set.

        Returns:
            BaseParser: The next parser that was set.
        """
        if not isinstance(next_parser, BaseParser):
            raise ValueError(
                f"next_parser {next_parser.__class__.__name__} must be a BaseParser"
            )
        self.next_parser = next_parser
        return self

    def parse(self, data: Any) -> Any:
        """
        Parse the input data and execute post-hooks.

        Args:
            data (Any): The input data to be parsed.

        Returns:
            Any: The parsed result.
        """
        try:
            result = self._parse(data)
            self._execute_post_hooks(result)
        except Exception as e: # pylint: disable=W0703
            logger.error("Parser [%s] error: %s", self.__class__.__name__, str(e))
            self._logger.error("Parser [%s] error: %s", self.__class__.__name__, str(e), exc_info=True)
            return data
        return result

    def register_post_hook(self, hook: Callable[[Any], None]) -> "BaseParser":
        """
        Register a post-hook to be executed after parsing.

        Args:
            hook (Callable[[Any], None]): The hook function to be registered.

        Returns:
            BaseParser: The current parser instance.

        Raises:
            ValueError: If the hook is not callable.
        """
        if callable(hook):
            self._post_hooks.append(hook)
        else:
            raise ValueError("Hook must be callable")
        return self

    def _execute_post_hooks(self, res: Any) -> None:
        """
        Execute all registered post-hooks asynchronously.

        Args:
            res (Any): The result to be passed to the post-hooks.
        """
        if not self._post_hooks:
            return

        processes = []
        for hook in self._post_hooks:
            p = Process(target=hook, args=(res,))
            p.start()
            hook_class = hook.__self__.__class__.__name__ if hasattr(hook, '__self__') else 'Unknown'
            hook_name = f"{hook_class}.{hook.__name__}"
            processes.append((p, hook_name))
            self._logger.info("Parser [%s] post hook [%s] start", self.__class__.__name__, hook_name)

        for p, hook_name in processes:
            try:
                p.join(timeout=self.EXEC_HOOK_TIMEOUT)
                if p.is_alive():
                    logger.error(
                        "Parser [%s] post hook [%s] timeout after %s seconds, terminating",
                        self.__class__.__name__, hook_name, self.EXEC_HOOK_TIMEOUT
                    )
                    p.terminate()
                    p.join()
                else:
                    self._logger.info(
                        "Parser [%s] post hook [%s] completed",
                        self.__class__.__name__, hook_name
                    )
            except Exception as e:  # pylint: disable=W0703
                self._logger.error(
                    "Parser [%s] post hook [%s] failed: %s",
                    self.__class__.__name__, hook_name, str(e), exc_info=True
                )
                if p.is_alive():
                    p.terminate()

    def _parse(self, data: Any) -> Any:
        """
        Abstract method to be implemented by subclasses for parsing logic.

        Args:
            data (Any): The input data to be parsed.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses should implement this!")


class DummyParser(BaseParser):
    """Dummy parser"""
    def _parse(self, data: Any) -> Any:
        """Dummy parse"""
        return data
