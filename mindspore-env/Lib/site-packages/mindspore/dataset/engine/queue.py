# Copyright 2021-2024 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
This dataset module creates an internal queue class to more optimally pass data
between multiple processes in Python.  It has same API as multiprocessing.queue
but it will pass large data through shared memory.
"""

import errno
import multiprocessing
import platform
import queue
import types

import dill
import numpy as np

from mindspore import log as logger
import mindspore._c_dataengine as cde
from ..transforms.py_transforms_util import ExceptionHandler


def get_total_size(data):
    """Calculate the total size of numpy arrays."""
    total_size = 0
    for column in data:
        if isinstance(column, np.ndarray):
            total_size += column.nbytes
    return total_size


class _SharedQueue(multiprocessing.queues.Queue):
    """
    Class to implement a queue using shared memory for better performance.
    Args:
        size: Number of elements in the queue.
        count: Shared variable to suppress log printing.
        copy_out: Whether to copy the data from shared memory to process virtual memory. Default: ``True``.
        max_rowsize: Maximum size of row in MB that is used for shared memory allocation to copy
            data between processes. If set to -1, shared memory will be dynamically allocated with
            the actual size of data. Default: -1.
    """

    def __init__(self, size, count, copy_out=True, max_rowsize=-1):
        super().__init__(size, ctx=multiprocessing.get_context())

        self.copy_out = copy_out

        # pipe can hold up to 65,636 bytes at a time
        # there is less benefit for small data. To small data it can be slower as we need to pass 100 bytes of metadata
        # and then access the shared memory.
        self.min_shared_mem = 10000
        self.data_immediate = 0
        self.data_shared = 1
        self.count = count
        self.print_error = True
        self.shm_list = []
        self.seg_pos = 0
        # num_seg has to be 2 more than the queue size. We can have remote worker filling a buffer, main process
        # reading a buffer and also have a full queue of buffers in the meta-data queue
        self.num_seg = size + 2

        self.dynamic_shm = True
        self.fd_list = []
        self.seg_size = 0
        if platform.system().lower() != 'windows' and max_rowsize == -1:
            self.dynamic_shm = True
            self.fd_list = []
        else:
            self.dynamic_shm = False
            # change max_rowsize in MB into bytes
            self.seg_size = max_rowsize * 1024 * 1024
            for _ in range(self.num_seg):
                try:
                    shared_array = multiprocessing.Array("b", self.seg_size)
                except OSError as e:
                    if e.errno == errno.ENOMEM:
                        raise RuntimeError("Failed to allocate shared memory for {0} elements of {1}MB: {2}"
                                           .format(self.num_seg, self.seg_size / 1024 / 1024, e))
                    raise
                else:
                    self.shm_list.append(shared_array)

    def __getstate__(self):
        copy_out = dill.dumps(self.copy_out)
        min_shared_mem = dill.dumps(self.min_shared_mem)
        data_immediate = dill.dumps(self.data_immediate)
        data_shared = dill.dumps(self.data_shared)
        # cannot dill 'mmap.mmap' object : self.count
        print_error = dill.dumps(self.print_error)
        shm_list = dill.dumps(self.shm_list)
        seg_pos = dill.dumps(self.seg_pos)
        num_seg = dill.dumps(self.num_seg)
        dynamic_shm = dill.dumps(self.dynamic_shm)
        fd_list = dill.dumps(self.fd_list)
        seg_size = dill.dumps(self.seg_size)
        # multiprocessing attribute
        _closed = dill.dumps(self._closed)
        super_variable = super().__getstate__()
        return (copy_out, min_shared_mem, data_immediate, data_shared, print_error, shm_list,
                seg_pos, num_seg, dynamic_shm, fd_list, seg_size, _closed, super_variable)

    def __setstate__(self, state):
        copy_out, min_shared_mem, data_immediate, data_shared, print_error, shm_list, \
            seg_pos, num_seg, dynamic_shm, fd_list, seg_size, _closed, super_variable = state
        self.copy_out = dill.loads(copy_out)
        self.min_shared_mem = dill.loads(min_shared_mem)
        self.data_immediate = dill.loads(data_immediate)
        self.data_shared = dill.loads(data_shared)
        self.print_error = dill.loads(print_error)
        self.shm_list = dill.loads(shm_list)
        self.seg_pos = dill.loads(seg_pos)
        self.num_seg = dill.loads(num_seg)
        self.dynamic_shm = dill.loads(dynamic_shm)
        self.fd_list = dill.loads(fd_list)
        self.seg_size = dill.loads(seg_size)
        self._closed = dill.loads(_closed)
        super().__setstate__(super_variable)

    def put_until(self, data, timeout=None, exit_signal=None):
        """Put data into the queue. Block until timeout is reached or exit_signal is set."""
        while True:
            try:
                self.put(data, timeout=timeout)
                return
            except queue.Full as e:
                if exit_signal is None:
                    raise e
                if exit_signal.is_set():
                    return
                continue

    def put(self, data, timeout=None):
        if isinstance(data, ExceptionHandler):  # pylint: disable=too-many-nested-blocks
            super().put(data, timeout=timeout)
        else:
            name_list = []
            start_bytes = 0
            if not isinstance(data, tuple):
                data = (data,)
            if isinstance(data, np.ndarray):
                name_list.append((self.data_immediate, np.array(data)))
            else:
                if self.dynamic_shm:
                    total_size = get_total_size(data)
                    if total_size > 0:
                        self.check_and_create_shm(total_size)
                for column in data:
                    # the map:pyfunc is a yield generator which can't be serialized
                    if isinstance(column, types.GeneratorType):
                        raise TypeError("Cannot pickle {} object, please verify pyfunc return with numpy array"
                                        .format(type(column)))
                    if self.dynamic_shm and isinstance(column, np.ndarray) and column.nbytes > 0:
                        shm = self.shm_list[self.seg_pos]
                        fd = self.fd_list[self.seg_pos]
                        dest = np.ndarray(column.shape, column.dtype, buffer=shm.buf(), offset=start_bytes)
                        np.copyto(dest, column)
                        start_bytes += column.nbytes
                        shm_metadata = (shm.name(), fd, total_size)
                        name_list.append((self.data_shared, self.seg_pos, column.dtype, column.shape, shm_metadata))
                    elif (isinstance(column, np.ndarray) and column.size > self.min_shared_mem
                          and start_bytes + column.nbytes < self.seg_size):
                        # need to convert start_bytes to offset in array
                        start_offset = start_bytes
                        shm = self.shm_list[self.seg_pos]
                        dest = np.ndarray(column.shape, column.dtype, buffer=shm.get_obj(), offset=start_offset)
                        np.copyto(dest, column)
                        byte = column.nbytes
                        byte = 8 * ((byte + 7) // 8)
                        start_bytes += byte
                        name_list.append((self.data_shared, self.seg_pos, byte, column.dtype, column.shape))
                    else:
                        if isinstance(column, np.ndarray) and column.size > self.min_shared_mem:
                            # Only print out error the first time it happens
                            if self.count.value == 0 and self.print_error:
                                logger.warning(
                                    "Using shared memory queue, but rowsize is larger than allocated memory "
                                    + "max_rowsize: "
                                    + str(self.seg_size / 1024 / 1024)
                                    + "MB, current rowsize: "
                                    + str((start_bytes + column.nbytes) / 1024 / 1024)
                                    + "MB."
                                )
                                self.print_error = False
                                self.count.value += 1
                        name_list.append((self.data_immediate, column))
            super().put(name_list, timeout=timeout)
            # note above could generate a queue full exception.  It will be handled by teh caller
            # only increment seg_pos after successfully adding to metadata queue

            if start_bytes > 0:
                self.seg_pos = (self.seg_pos + 1) % self.num_seg

    def get_until(self, timeout=None, exit_signal=None):
        """Get data from the queue. Block until timeout is reached or exit_signal is set."""
        while True:
            try:
                result = self.get(timeout=timeout)
            except queue.Empty as e:
                if exit_signal is None:
                    raise e
                if exit_signal.is_set():
                    return None
                continue
            if result is None:
                # receive finish signal
                return None
            if exit_signal.is_set():
                # loop until the queue becomes empty
                continue
            return result

    def get(self, timeout=None):
        raw_data = super().get(timeout=timeout)
        if isinstance(raw_data, ExceptionHandler):
            return raw_data
        result = []
        start_bytes = 0
        for column in raw_data:
            if column[0] == self.data_shared:
                if self.dynamic_shm:
                    seg_pos, dtype, shape, shm_metadata = column[1:]
                    if start_bytes == 0:
                        # only need to check once since all the columns are stored in the same shared memory
                        self.check_and_attach_shm(seg_pos, shm_metadata)
                    shm = self.shm_list[seg_pos]
                    array = np.ndarray(shape, dtype, buffer=shm.buf(), offset=start_bytes)
                    start_bytes += array.nbytes
                else:
                    seg_pos, byte, dtype, shape = column[1:]
                    start_offset = start_bytes
                    shm = self.shm_list[seg_pos]
                    array = np.ndarray(shape, dtype, buffer=shm.get_obj(), offset=start_offset)
                    start_bytes += byte
                if self.copy_out:
                    result.append(np.copy(array))
                else:
                    result.append(array)
            elif column[0] == self.data_immediate:
                result.append(column[1])
            else:
                raise RuntimeError("SharedQueue, invalid entry in metadata.")
        return tuple(result)

    def check_and_create_shm(self, size):
        """Check if the shared memory is initialized and of sufficient size."""
        if len(self.shm_list) == self.seg_pos:
            # shared memory has not been created and appended to the cache list
            shm = cde.SharedMemory(None, True, -1, size)
            shared_fd = multiprocessing.reduction.DupFd(shm.fd())
            self.shm_list.append(shm)
            self.fd_list.append(shared_fd)
        elif len(self.shm_list) > self.seg_pos:
            if self.shm_list[self.seg_pos].size() < size:
                # shared memory is not big enough to hold the data
                shm = cde.SharedMemory(None, True, -1, size)
                shared_fd = multiprocessing.reduction.DupFd(shm.fd())
                self.shm_list[self.seg_pos] = shm
                self.fd_list[self.seg_pos] = shared_fd
        else:
            raise RuntimeError("The shared memory index is larger than the length of shared memory list. "
                               "Uninitialized shared memory may exist.")

    def check_and_attach_shm(self, shm_index, shm_metadata):
        """Check if the shared memory is initialized and is the same as the current one."""
        shm_name, fd, size = shm_metadata
        if len(self.shm_list) == shm_index:
            # shared memory has not been created and appended to the cache list
            fd = fd.detach()
            shm = cde.SharedMemory(shm_name, False, fd, size)
            self.shm_list.append(shm)
            self.fd_list.append(fd)
        elif len(self.shm_list) > shm_index:
            if self.shm_list[shm_index].name() != shm_name:
                # shared memory has changed
                fd = fd.detach()
                shm = cde.SharedMemory(shm_name, False, fd, size)
                self.shm_list[shm_index] = shm
                self.fd_list[shm_index] = fd
        else:
            raise RuntimeError("The shared memory index is larger than the length of shared memory list. "
                               "Uninitialized shared memory may exist.")

    def __del__(self):
        if not self.dynamic_shm:
            shm_list_len = len(self.shm_list)
            for idx in range(shm_list_len):
                del self.shm_list[shm_list_len - idx - 1]
            self.shm_list.clear()
            del self.shm_list

        self.close()
        self.join_thread()


class _Queue(multiprocessing.queues.Queue):
    """Specialized multiprocessing Queue that supports interrupted operations."""

    def __init__(self, size):
        super().__init__(size, ctx=multiprocessing.get_context())

    def put_until(self, data, timeout=None, exit_signal=None):
        """Put data into the queue. Block until timeout is reached or exit_signal is set."""
        while True:
            try:
                self.put(data, timeout=timeout)
                return
            except queue.Full as e:
                if exit_signal is None:
                    raise e
                if exit_signal.is_set():
                    return
                continue

    def get_until(self, timeout=None, exit_signal=None):
        """Get data from the queue. Block until timeout is reached or exit_signal is set."""
        while True:
            try:
                r = self.get(timeout=timeout)
            except queue.Empty as e:
                if exit_signal is None:
                    raise e
                if exit_signal.is_set():
                    return None
                continue
            if r is None:
                # receive finish signal
                return None
            if exit_signal.is_set():
                # loop until the queue becomes empty
                continue
            return r
