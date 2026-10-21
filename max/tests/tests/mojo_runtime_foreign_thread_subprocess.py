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
"""Subprocess entry: call a CompilerRT function from ``asyncio.to_thread`` after ``InferenceSession``.

Regression test for GEX-3592: ``Runtime::getCurrentRuntimeOrNull()`` returned
null for threads not managed by AsyncRT, causing CompilerRT functions that
dereference it to segfault.

The test directly calls ``KGEN_CompilerRT_AsyncRT_ParallelismLevel`` from a
foreign thread.  That function unconditionally dereferences the result of
``getCurrentRuntimeOrNull()``:

    auto rt = Runtime::getCurrentRuntimeOrNull();
    return rt->getWorkQueue()->getParallelismLevel();

Executed via ``python …/mojo_runtime_foreign_thread_subprocess.py`` from
``test_mojo_runtime_foreign_thread`` so the interpreter has no prior
MAX/Mojo state from pytest.
"""

from __future__ import annotations

import asyncio
import ctypes
import os

from max.driver import CPU
from max.engine import InferenceSession


def _load_compiler_rt() -> ctypes.CDLL:
    """Return a ctypes handle to the already-loaded KGEN CompilerRT library.

    Imports of ``max`` load ``libKGENCompilerRTShared.so`` into the process.
    We locate it via ``/proc/self/maps`` and open it with ctypes (which on
    Linux just increments the existing ``dlopen`` reference count and returns
    the same handle — no re-initialisation).
    """
    with open("/proc/self/maps") as f:
        maps = f.read()
    for line in maps.splitlines():
        if "libKGENCompilerRTShared.so" in line:
            path = line.split()[-1]
            if os.path.isfile(path):
                return ctypes.CDLL(path)
    raise RuntimeError(
        "libKGENCompilerRTShared.so not found in /proc/self/maps — "
        "was InferenceSession imported?"
    )


async def _async_main() -> None:
    # Initialise InferenceSession on the event-loop thread.  This creates the
    # global AsyncRT runtime and sets TLS for this thread.
    _session = InferenceSession(devices=[CPU()])

    compiler_rt = _load_compiler_rt()
    parallelism_level_fn = compiler_rt.KGEN_CompilerRT_AsyncRT_ParallelismLevel
    parallelism_level_fn.restype = ctypes.c_uint32

    def call_compiler_rt_on_foreign_thread() -> None:
        # This thread has no AsyncRT TLS.  Before the fix,
        # KGEN_CompilerRT_AsyncRT_ParallelismLevel would dereference a null
        # Runtime pointer and segfault.
        level = parallelism_level_fn()
        assert level > 0, f"expected positive parallelism level, got {level}"

    await asyncio.to_thread(call_compiler_rt_on_foreign_thread)


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
