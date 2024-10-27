# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from memory import UnsafePointer
from compile import _internal_compile_code, compile_info
from testing import *
from gpu import *
from gpu.host import *
from gpu.memory import AddressSpace

alias target_short_ptr = __mlir_attr[
    `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
    `arch = "sm_80", `,
    `features = "+ptx81", `,
    `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
    `simd_bit_width = 128,`,
    `index_bit_width = 64`,
    `> : !kgen.target`,
]

alias target_regular = __mlir_attr[
    `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
    `arch = "sm_80", `,
    `features = "+ptx81", `,
    `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
    `simd_bit_width = 128,`,
    `index_bit_width = 64`,
    `> : !kgen.target`,
]


def test_data_layout_llvm[emission_kind: StringLiteral]():
    fn my_func(src: UnsafePointer[Int32]):
        return

    var target_short_llvm = _internal_compile_code[
        my_func, emission_kind=emission_kind, target=target_short_ptr
    ]()
    var target_regular_llvm = _internal_compile_code[
        my_func, emission_kind=emission_kind, target=target_regular
    ]()

    assert_true(
        "e-p3:32:32-p4:32:32-p5:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
        in target_short_llvm
    )

    assert_true(
        "e-i64:64-i128:128-v16:16-v32:32-n16:32:64" in target_regular_llvm
    )


def test_data_layout_asm():
    fn my_func(src: UnsafePointer[Int32]):
        var a = stack_allocation[
            20, Int32, address_space = AddressSpace.SHARED
        ]()
        a[ThreadIdx.x()] = src[0]
        barrier()

    var target_short_asm = _internal_compile_code[
        my_func, emission_kind="asm", target=target_short_ptr
    ]()
    var target_regular_asm = _internal_compile_code[
        my_func, emission_kind="asm", target=target_regular
    ]()
    assert_false(target_short_asm == target_regular_asm)


def main():
    test_data_layout_llvm["llvm"]()
    test_data_layout_llvm["llvm-opt"]()
    # TODO: Uncommend after figuring out MOCO-1390
    # test_data_layout_asm()
