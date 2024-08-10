# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# COM: Disabled temporarily as the followup PR fixes this.
# REQUIRES: DISABLED
# RUN: %mojo-no-debug %s

from gpu.host._compile import _compile_code, _get_nvptx_target
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from testing import *


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
            register_intrinsics, target = _get_nvptx_target["sm_90"]()
        ]().asm
    )
    assert_true("setmaxnreg.inc.sync.aligned.u32" in asm)
    assert_true("setmaxnreg.dec.sync.aligned.u32" in asm)


def main():
    test_register_intrinsics_sm80()
    test_register_intrinsics_sm90()
