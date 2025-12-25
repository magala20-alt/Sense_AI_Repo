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
"""Process bar."""
import os
import sys
import time
from typing import Iterable, Optional, Any
import threading


class ProcessBar:
    """
    A progress bar for tracking the progress of an iterable or a process with a known total.
    """
    BLANK_SPACE_NUM = 20
    FINISH_TEXT = "Done"

    def __init__(
            self,
            iterable: Optional[Iterable] = None,
            desc: str = "",
            bar_length: int = 20,
            update_interval: float = 1.0,
    ):
        """
        Initialize the ProcessBar.

        Args:
            iterable: An optional iterable to track progress.
            desc: A description of the process being tracked.
            bar_length: The length of the progress bar in characters.
            update_interval: The minimum time interval between progress bar updates.
        """
        if not isinstance(iterable, Iterable):
            raise ValueError("Must provide an iterable")

        if not isinstance(desc, str):
            raise ValueError("desc must be a string")

        if not isinstance(bar_length, int):
            raise ValueError("bar_length must be an integer")

        if bar_length <= 0:
            raise ValueError("bar_length must be greater than 0")

        if not isinstance(update_interval, float):
            raise ValueError("update_interval must be a float")

        if update_interval < 0:
            raise ValueError("update_interval must be greater than 0")

        self.iterable: Iterable = iterable
        self.total: int = len(iterable)
        self.desc: str = f"[{os.getpid()}] {desc}"
        self.bar_length: int = bar_length
        self.update_interval: float = update_interval
        self.current: int = 0
        self.cur_item_name: Optional[str] = None
        self.start_time: float = time.time()
        self.last_update_time: float = self.start_time
        self._stop_refresh = False
        self._refresh_thread = None
        self._start_auto_refresh()

    def _start_auto_refresh(self) -> None:
        """
        Start auto refresh thread.
        """
        def refresh_loop():
            while not self._stop_refresh:
                if self.current > 0:
                    self._print_progress(time.time())
                time.sleep(self.update_interval)

        self._refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        self._refresh_thread.start()

    def update(self, n: int = 1, item_name: Optional[str] = None) -> None:
        """
        Update the progress bar.

        Args:
            n: The number of items or steps to increment the progress by.
            item_name: The name of the current item being processed.
        """
        self.current += n
        self.cur_item_name = item_name

    def _print_progress(self, now: float) -> None:
        """
        Print the current progress to the console.

        Args:
            now: The current timestamp.
        """
        elapsed = now - self.start_time
        if self.total > 0:
            progress = min(1, self.current / self.total)
            block = int(round(self.bar_length * progress))

            text = (
                f"\r{self.desc}: [{block * '#' + (self.bar_length - block) * ' '}] "
                f"{self.current}/{self.total} {self.cur_item_name} "
                f"Elapsed: {int(elapsed)}s"
            )
            # 添加额外的空格和回车来清除可能的残留字符
            text = text + ' ' * self.BLANK_SPACE_NUM + '\r'
            sys.stdout.write(text)
            sys.stdout.flush()

    def __iter__(self) -> Iterable[Any]:
        """
        Iterate over the items in the iterable, updating the progress bar for each item.

        Yields:
            The next item from the iterable.

        Raises:
            ValueError: If no iterable was provided during initialization.
        """
        if self.iterable is None:
            raise ValueError("Must provide an iterable")
        try:
            iterator = iter(self.iterable)
            # 预先获取第一个元素
            try:
                first_item = next(iterator)
            except StopIteration:
                return

            # 初始化显示，显示0/total和第一个元素的名称
            self.cur_item_name = first_item.__class__.__name__
            self._print_progress(time.time())

            # 处理第一个元素
            yield first_item
            self.update(item_name=first_item.__class__.__name__)
            self._print_progress(time.time())

            # 处理剩余元素
            for item in iterator:
                # 先更新当前要处理的元素名称
                self.cur_item_name = item.__class__.__name__
                self._print_progress(time.time())
                yield item
                self.update()

            # 显示完成状态
            self.cur_item_name = self.FINISH_TEXT
            self._print_progress(time.time())
            sys.stdout.write("\n")

        finally:
            self._stop_refresh = True
            if self._refresh_thread:
                self._refresh_thread.join()
