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
"""Compiles the split Mamba-2 SSD scan for CUDA and HIP targets.

The wrapper dispatches the split kernel on every CUDA and HIP GPU, but CI
only runs it on some of them. This cross-compiles it for the others so a
codegen failure shows up here.
"""

from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.compile import _compile_code
from layout import TileTensor, row_major
from state_space.mamba2_ssd_scan import (
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split,
)
from std.testing import TestSuite, assert_true


def compile_split[
    target_name: StaticString, entry_marker: StaticString
](ctx: DeviceContext) raises:
    # The shapes only fix the layout types; nothing is launched.
    var x_d = ctx.enqueue_create_buffer[.bfloat16](1)
    var i32_d = ctx.enqueue_create_buffer[.int32](1)
    var bool_d = ctx.enqueue_create_buffer[.bool](1)
    var u32_d = ctx.enqueue_create_buffer[.uint32](1)
    var pool_d = ctx.enqueue_create_buffer[.float32](1)
    comptime T3 = type_of(TileTensor(x_d, row_major(1, 1, 1)))
    comptime T2 = type_of(TileTensor(x_d, row_major(1, 1)))
    comptime T1 = type_of(TileTensor(x_d, row_major(1)))
    comptime Pool = type_of(TileTensor(pool_d, row_major(1, 1, 1, 1)))
    comptime Qsl = type_of(TileTensor(i32_d, row_major(1)))
    comptime His = type_of(TileTensor(bool_d, row_major(1)))
    comptime Slots = type_of(TileTensor(u32_d, row_major(1)))

    var asm = _compile_code[
        mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split[
            .bfloat16,
            128,
            T3.LayoutType,
            T2.LayoutType,
            T1.LayoutType,
            T3.LayoutType,
            T3.LayoutType,
            T1.LayoutType,
            T1.LayoutType,
            T3.LayoutType,
            Pool.LayoutType,
            Qsl.LayoutType,
            His.LayoutType,
            Slots.LayoutType,
            T3.Engine,
            8,
        ],
        target=get_gpu_target[target_name](),
    ]()
    assert_true(entry_marker in asm, "no kernel entry for " + target_name)


def test_compile_split_nvidia() raises:
    with DeviceContext() as ctx:
        compile_split["sm_90a", ".entry"](ctx)
        compile_split["sm_100a", ".entry"](ctx)
        compile_split["sm_103a", ".entry"](ctx)


def test_compile_split_amd() raises:
    with DeviceContext() as ctx:
        compile_split["mi300x", ".amdhsa_kernel"](ctx)
        compile_split["mi355x", ".amdhsa_kernel"](ctx)
        compile_split["mi455x", ".amdhsa_kernel"](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
