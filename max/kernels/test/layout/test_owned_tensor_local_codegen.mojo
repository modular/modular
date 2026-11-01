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
"""Tests that a `LOCAL` `OwnedGPUTensor` is promoted to registers in PTX.

A local `stack_allocation` starts out in the PTX local-memory depot. Once every
access has a compile-time offset, LLVM should promote the whole tensor to
registers and leave no `.local` traffic behind.
"""

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from std.testing import assert_false, assert_true, TestSuite

from layout import TileTensor, row_major
from layout.owned_tensor import OwnedGPUTensor

# Target SM80 (Ampere) explicitly so the test cross-compiles to NVIDIA PTX
# regardless of the host GPU.
comptime _SM80 = get_gpu_target["sm_80"]()

comptime _ROWS = 4
comptime _COLS = 4


def static_index_kernel(
    src: Pointer[Float32, ImmutAnyOrigin], dst: Pointer[Float32, MutAnyOrigin]
):
    var owned = OwnedGPUTensor[.float32, .LOCAL](row_major[_ROWS, _COLS]())
    var tile = TileTensor(owned)
    # Promotion needs constant offsets, and LLVM leaves runtime loops over
    # these extents rolled, so unroll at compile time.
    comptime for row in range(_ROWS):
        comptime for col in range(_COLS):
            tile[row, col] = src.unsafe_load(row * _COLS + col)
    # Read back transposed so the stores can't be forwarded straight through.
    comptime for row in range(_ROWS):
        comptime for col in range(_COLS):
            dst.unsafe_store(row * _COLS + col, tile[col, row][0])


def dynamic_index_kernel(
    src: Pointer[Float32, ImmutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    row: Int,
):
    var owned = OwnedGPUTensor[.float32, .LOCAL](row_major[_ROWS, _COLS]())
    var tile = TileTensor(owned)
    for r in range(_ROWS):
        for c in range(_COLS):
            tile[r, c] = src.unsafe_load(r * _COLS + c)
    dst.unsafe_store(0, tile[row, 0][0])


def test_local_tensor_lowers_to_registers() raises:
    var asm = _compile_code[static_index_kernel, target=_SM80]()
    assert_false(
        "__local_depot" in asm, "local OwnedGPUTensor kept a local-memory depot"
    )
    assert_false("ld.local" in asm, "local OwnedGPUTensor emitted ld.local")
    assert_false("st.local" in asm, "local OwnedGPUTensor emitted st.local")


def test_dynamic_index_keeps_local_memory() raises:
    """The control arm: a runtime index can't be promoted, so `.local` stays.

    Without this the test above could pass against an engine that never put
    the tensor in local memory at all, or against PTX whose spelling of local
    memory the checks no longer match.
    """
    var asm = _compile_code[dynamic_index_kernel, target=_SM80]()
    assert_true(
        "__local_depot" in asm,
        "expected a local-memory depot for a dynamically indexed tensor",
    )
    assert_true(
        "ld.local" in asm,
        "expected ld.local for a dynamically indexed tensor",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
