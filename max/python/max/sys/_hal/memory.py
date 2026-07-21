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
"""Module-level synchronous memory ops: ``copy``, ``set_memory``, ``fill``."""

from __future__ import annotations

from .buffer import Buffer
from .mojo_module import (  # type: ignore[import-not-found]
    copy as _mojo_copy,
)
from .mojo_module import (
    fill as _mojo_fill,
)
from .mojo_module import (
    set_memory as _mojo_set_memory,
)


def copy(*, dst: Buffer, src: Buffer) -> None:
    """Synchronously copies buffer ``src`` into the front of buffer ``dst``.

    Transfers exactly ``src``'s bytes; ``dst`` must be at least that large.
    The transport is chosen from each operand's residency (device-to-device,
    or a pinned buffer to/from device); a pinned-to-pinned copy is performed
    host-side. Blocks until the transfer completes. Dispatch and the blocking,
    stream-less plugin copy ops all run on the Mojo side — no queue is created.
    """
    _mojo_copy(dst._inner, src._inner)


def set_memory(*, dst: Buffer, value: int) -> None:
    """Synchronously sets every byte of ``dst`` to ``value``.

    Sets all of ``dst``'s bytes and blocks until the fill completes; no queue is
    created. The blocking, stream-less plugin op runs on the Mojo side.
    """
    _mojo_set_memory(dst._inner, value)


def fill(*, dst: Buffer, value: int, value_size: int) -> None:
    """Synchronously fills ``dst`` with a repeated ``value_size``-byte value.

    Fills all of ``dst``'s bytes and blocks until the fill completes; no queue
    is created. ``dst``'s byte size must be a multiple of ``value_size``, which
    must be one of 1, 2, 4, or 8; a ``value_size`` of 1 is equivalent to
    :func:`set_memory`. The blocking, stream-less plugin op runs on the Mojo
    side.
    """
    _mojo_fill(dst._inner, value, value_size)
