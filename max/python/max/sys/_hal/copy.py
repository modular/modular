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
"""``copy`` — module-level synchronous copy between HAL buffers."""

from __future__ import annotations

from .buffer import Buffer
from .mojo_module import copy as _mojo_copy  # type: ignore[import-not-found]


def copy(*, dst: Buffer, src: Buffer) -> None:
    """Synchronously copies buffer ``src`` into the front of buffer ``dst``.

    Transfers exactly ``src``'s bytes; ``dst`` must be at least that large.
    The transport is chosen from each operand's residency (device-to-device,
    or a pinned buffer to/from device); a pinned-to-pinned copy is performed
    host-side. Blocks until the transfer completes. Dispatch and the blocking,
    stream-less plugin copy ops all run on the Mojo side — no queue is created.
    """
    _mojo_copy(dst._inner, src._inner)
