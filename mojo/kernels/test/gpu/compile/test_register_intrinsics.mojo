# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host._compile import _compile_code, _get_nvptx_target
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from testing import *


@always_inline
fn _get_nvptx_target_sm90() -> __mlir_type.`!kgen.target`:
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_90", `,
        `features = "+ptx81", `,
        `data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",`,
        `simd_bit_width = 128> : !kgen.target`,
    ]


fn register_intrinsics():
    warpgroup_reg_alloc[42]()
    warpgroup_reg_dealloc[42]()


def test_register_intrinsics_sm80():
    alias asm = str(
        _compile_code[register_intrinsics, target = _get_nvptx_target()]().asm
    )
    assert_false("setmaxnreg.inc.sync.aligned.u32" in asm)
    assert_false("setmaxnreg.dec.sync.aligned.u32" in asm)


def test_register_intrinsics_sm90():
    alias asm = str(
        _compile_code[
            register_intrinsics, target = _get_nvptx_target_sm90()
        ]().asm
    )
    assert_true("setmaxnreg.inc.sync.aligned.u32" in asm)
    assert_true("setmaxnreg.dec.sync.aligned.u32" in asm)


def main():
    test_register_intrinsics_sm80()
    test_register_intrinsics_sm90()
