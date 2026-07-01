# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""MAX HAL — Python projection of the Mojo HAL.

Mirrors the Mojo-side ``std.sys._hal`` namespace: ``Driver`` is the
entry point (configured via ``MODULAR_DRIVER_PLUGINS``), and downstream
types are obtained through the lifecycle chain
``Driver -> Device -> Context -> Queue / Stream -> Event``.
"""

from __future__ import annotations

import mojo.importer

from .buffer import Buffer, BufferView
from .bundle import Bundle
from .context import Context
from .device import Device
from .driver import Driver
from .event import Event
from .function import Function
from .queue import Queue
from .stream import Stream

__all__ = [
    "Buffer",
    "BufferView",
    "Bundle",
    "Context",
    "Device",
    "Driver",
    "Event",
    "Function",
    "Queue",
    "Stream",
]
