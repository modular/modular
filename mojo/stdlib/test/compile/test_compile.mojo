# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from compile import compile_info
from gpu import *
from gpu.host import *
from gpu.memory import AddressSpace
from memory import stack_allocation
from testing import *


def test_compile_llvm():
    @parameter
    fn my_add_function[
        dtype: DType, size: Int
    ](x: SIMD[dtype, size], y: SIMD[dtype, size]) -> SIMD[dtype, size]:
        return x + y

    alias func = my_add_function[DType.float32, 4]
    assert_true("fadd" in compile_info[func, emission_kind="llvm"]())


alias target_short_ptr = __mlir_attr[
    `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
    `arch = "sm_80", `,
    `features = "+ptx81", `,
    `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
    `simd_bit_width = 128,`,
    `index_bit_width = 64`,
    `> : !kgen.target`,
]

alias target_regular = __mlir_attr[
    `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
    `arch = "sm_80", `,
    `features = "+ptx81", `,
    `data_layout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
    `simd_bit_width = 128,`,
    `index_bit_width = 64`,
    `> : !kgen.target`,
]


def test_data_layout_llvm[emission_kind: StaticString]():
    fn my_func(src: UnsafePointer[Int32]):
        return

    var target_short_llvm = compile_info[
        my_func, emission_kind=emission_kind, target=target_short_ptr
    ]()
    var target_regular_llvm = compile_info[
        my_func, emission_kind=emission_kind, target=target_regular
    ]()

    assert_true(
        "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
        in target_short_llvm
    )

    assert_true(
        "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
        in target_regular_llvm
    )


def test_data_layout_asm():
    fn my_func(src: UnsafePointer[Int32]):
        var a = stack_allocation[
            20, Int32, address_space = AddressSpace.SHARED
        ]()
        a[thread_idx.x] = src[0]
        barrier()

    var target_short_asm = compile_info[
        my_func,
        emission_kind="asm",
        compile_options="nvptx-short-ptr=true",
        target=target_short_ptr,
    ]()

    assert_true("mov.u32" in target_short_asm)
    assert_false("mov.u64" in target_short_asm)


def main():
    test_compile_llvm()
    test_data_layout_llvm["llvm"]()
    test_data_layout_llvm["llvm-opt"]()
    test_data_layout_asm()
