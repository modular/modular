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

from __future__ import annotations

import os
import sys

from max.driver import DLPackArray
from max.experimental.tensor import NestedArray, Number, Tensor
from python.runfiles import runfiles

# Point the eager warm cache at the build-time-warmed dir before the
# import-time precompile adopts it (see test_interpreter_ops.py).
_warm_rloc = os.environ.get("XARCH_WARM_RLOCATION")
if _warm_rloc:
    _runfiles = runfiles.Create()
    _resolved = _runfiles.Rlocation(_warm_rloc) if _runfiles else None
    if _resolved:
        os.environ["MODULAR_DERIVED_PATH"] = _resolved
    else:
        # Surface a miss: otherwise the warm silently won't adopt and the
        # cold-compile just reads as a timeout.
        print(
            f"[eager-warm] XARCH_WARM_RLOCATION={_warm_rloc!r} did not resolve; "
            "warm cache not adopted -- GC sweep will cold-compile.",
            file=sys.stderr,
            flush=True,
        )


def assert_all_close(
    t1: DLPackArray | NestedArray | Number,
    t2: Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    if not isinstance(t1, Tensor):
        t1 = Tensor(t1, dtype=t2.dtype, device=t2.device)

    absolute_difference = abs(t1 - t2)
    # TODO: div0
    left_relative_difference = abs(absolute_difference / t1)
    right_relative_difference = abs(absolute_difference / t2)

    if (d := absolute_difference.max()) > atol:
        idx = absolute_difference.argmax().item()
        raise AssertionError(
            f"atol: tensors not close at index {idx}, {d.item()} > {atol}: \n"
            f"   left[{idx}] = {t1[idx].item()}\n"
            f"  right[{idx}] = {t2[idx].item()}\n"
        )
    elif (d := left_relative_difference.max()) > rtol:
        idx = left_relative_difference.argmax().item()
        raise AssertionError(
            f"rtol: tensors not close at index {idx}, {d.item()} > {rtol}: \n"
            f"   left[{idx}] = {t1[idx].item()}\n"
            f"  right[{idx}] = {t2[idx].item()}\n"
        )
    elif (d := right_relative_difference.max()) > rtol:
        idx = right_relative_difference.argmax().item()
        raise AssertionError(
            f"rtol: tensors not close at index {idx}, {d.item()} > {rtol}: \n"
            f"   left[{idx}] = {t1[idx].item()}\n"
            f"  right[{idx}] = {t2[idx].item()}\n"
        )
