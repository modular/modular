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

"""Import-safety tests for ``max._distributed_ops``.

These run without a GPU: they simulate a host that *has* an accelerator whose
GPU architecture the broadcast kernel cannot compile for — the case where the
kernel JIT raises ``ImportError`` at module load — and assert the module still
imports and degrades gracefully.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import ModuleType

import max._distributed_ops as dist_ops
import pytest
from max import driver
from max.driver import CPU, Buffer
from max.dtype import DType


@contextmanager
def _simulate_unsupported_accelerator() -> Iterator[None]:
    """Reload ``max._distributed_ops`` as if a GPU is present but the broadcast
    kernel fails to compile for its architecture.

    Forces the ``accelerator_count() > 0`` branch to run on any host, then
    makes the ``from .distributed_ops import broadcast_kernel`` JIT import raise
    ``ImportError`` the way an unsupported GPU architecture does. The module is
    reloaded to its real state on exit.
    """
    real_import = builtins.__import__
    real_platform = sys.platform
    real_count = driver.accelerator_count

    def _failing_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if level == 1 and "broadcast_kernel" in fromlist:
            raise ImportError(
                "Import of Mojo module failed due to compilation error. "
                "note: constraint failed: GPU architecture 'fake' is not "
                "supported."
            )
        return real_import(name, globals, locals, fromlist, level)

    try:
        sys.platform = "linux"
        driver.accelerator_count = lambda: 1
        builtins.__import__ = _failing_import
        importlib.reload(dist_ops)
        yield
    finally:
        builtins.__import__ = real_import
        sys.platform = real_platform
        driver.accelerator_count = real_count
        # Restore the module to its real state for any later test in-process.
        importlib.reload(dist_ops)


def test_import_degrades_when_kernel_unavailable() -> None:
    with _simulate_unsupported_accelerator():
        assert dist_ops._broadcast_kernel is None


def test_call_without_kernel_raises_runtime_error() -> None:
    # CPU buffers that pass distributed_broadcast's argument validation so the
    # call reaches the missing-kernel guard; the kernel is never invoked. The
    # root output aliases the input, which the API permits.
    cpu = CPU()
    in_buf = Buffer.zeros(shape=[8], dtype=DType.uint8, device=cpu)
    out_buf = Buffer.zeros(shape=[8], dtype=DType.uint8, device=cpu)
    signals = [
        Buffer.zeros(shape=[8], dtype=DType.uint8, device=cpu) for _ in range(2)
    ]
    devices = [in_buf.device, out_buf.device]

    with _simulate_unsupported_accelerator():
        assert dist_ops._broadcast_kernel is None
        with pytest.raises(RuntimeError, match="distributed_broadcast"):
            dist_ops.distributed_broadcast(
                input_buffer=in_buf,
                output_buffers=[in_buf, out_buf],
                signal_buffers=signals,
                devices=devices,
                root=0,
            )
