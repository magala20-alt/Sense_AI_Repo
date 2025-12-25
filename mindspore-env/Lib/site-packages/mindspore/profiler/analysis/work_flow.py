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
"""Workflow management for parsers"""
from typing import Iterator, Optional
from mindspore.profiler.analysis.parser.base_parser import BaseParser


class WorkFlow:
    """
    Manages a chain of parsers.
    """
    def __init__(self, head_parser: Optional[BaseParser] = None):
        """
        Initialize WorkFlow with an optional head parser.

        Args:
            head_parser (Optional[BaseParser]): The first parser in the chain.
        """
        self.head_parser = head_parser

    def add_parser(self, parser: BaseParser) -> None:
        """
        Add a parser to the end of the chain.

        Args:
            parser (BaseParser): Parser to be added to the chain.
        """
        if not self.head_parser:
            self.head_parser = parser
            return

        current = self.head_parser
        while current.next_parser:
            current = current.next_parser
        current.next_parser = parser

    def __iter__(self) -> Iterator[BaseParser]:
        """
        Iterate through the chain of parsers.

        Yields:
            BaseParser: Each parser in the chain.
        """
        current = self.head_parser
        while current:
            yield current
            current = current.next_parser

    def __len__(self) -> int:
        """
        Get the length of the parser chain.

        Returns:
            int: The number of parsers in the chain.
        """
        length = 0
        current = self.head_parser
        while current:
            length += 1
            current = current.next_parser
        return length
