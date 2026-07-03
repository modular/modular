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
"""Hardware Abstraction Layer (HAL) — Mojo bindings.

Provides a Mojo-native library layer around the HAL C plugin API.
"""

from .status import HALError, STATUS_SUCCESS
from .plugin import FunctionHandle, RawDriver
from .driver import Driver, get_device_spec
from .device import Device
from .buffer import Buffer, BufferView
from .context import Context, RuntimeBundle
from .event import Event
from .queue import Queue
from .stream import Stream
