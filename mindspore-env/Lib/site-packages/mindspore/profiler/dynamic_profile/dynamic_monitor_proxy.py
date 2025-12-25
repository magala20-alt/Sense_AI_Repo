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
"""Dynamic Profiler utils"""
from mindspore.profiler.common.singleton import Singleton


@Singleton
class MsDynamicMonitorProxySingleton:
    """
    Class for dyno monitor proxy.
    """
    def __init__(self):
        self._proxy = None
        self._load_success = True

    def _load_proxy(self):
        if not self._proxy and self._load_success:
            try:
                from IPCMonitor import PyDynamicMonitorProxy
            except ImportError:
                self._load_success = False
                return
            self._proxy = PyDynamicMonitorProxy()

    def get_proxy(self):
        self._load_proxy()
        return self._proxy
